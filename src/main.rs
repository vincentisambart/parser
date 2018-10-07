// TODO:
// - For testing pragmas, have a look at clang's test/Sema/pragma-align-packed.c
// - Move tests to one (or multiple) other files
// - Add position to declarations
// - Make most panics/expect normal errors
// - Storage class (including typedef), function specifier
// - struct/union
// - Function definition
// - Variable initialization

mod error;
mod failable;
mod lex;
mod peeking;

use crate::error::{ParseError, ParseErrorKind};
use crate::failable::{FailableIterator, FailablePeekable};
use crate::lex::{Keyword, Position, PositionedToken, Punctuator, Token, TokenIter};

use bitflags::bitflags;
use std::collections::{HashMap, HashSet};

trait PeekingToken {
    type Error;

    fn advance_if_kw(&mut self, kw: Keyword) -> Result<bool, Self::Error>;
    fn advance_if_punc(&mut self, punc: Punctuator) -> Result<bool, Self::Error>;
    fn next_if_any_ident(&mut self) -> Result<Option<(String, Position)>, Self::Error>;
    fn next_if_any_kw(&mut self) -> Result<Option<(Keyword, Position)>, Self::Error>;
}

impl<I> PeekingToken for FailablePeekable<I>
where
    I: FailableIterator<Item = PositionedToken>,
{
    type Error = I::Error;

    fn advance_if_kw(&mut self, kw: Keyword) -> Result<bool, Self::Error> {
        self.advance_if(|token| match token {
            PositionedToken(Token::Keyword(x), _) if *x == kw => true,
            _ => false,
        })
    }

    fn advance_if_punc(&mut self, punc: Punctuator) -> Result<bool, Self::Error> {
        self.advance_if(|token| match token {
            PositionedToken(Token::Punctuator(x), _) if *x == punc => true,
            _ => false,
        })
    }

    fn next_if_any_ident(&mut self) -> Result<Option<(String, Position)>, I::Error> {
        let token = self.next_if(|token| match token {
            PositionedToken(Token::Identifier(_), _) => true,
            _ => false,
        })?;
        match token {
            Some(PositionedToken(Token::Identifier(ident), pos)) => Ok(Some((ident, pos))),
            Some(_) => unreachable!(),
            None => Ok(None),
        }
    }

    fn next_if_any_kw(&mut self) -> Result<Option<(Keyword, Position)>, I::Error> {
        let token = self.next_if(|token| match token {
            PositionedToken(Token::Keyword(_), _) => true,
            _ => false,
        })?;
        match token {
            Some(PositionedToken(Token::Keyword(kw), pos)) => Ok(Some((kw, pos))),
            Some(_) => unreachable!(),
            None => Ok(None),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BasicType {
    Void, // The standard doesn't consider "void" a basic type but that makes things easier
    Char,
    SignedChar,
    UnsignedChar,
    Short,
    UnsignedShort,
    Int,
    SignedInt, // Could be handled differently from Int in bit fields
    UnsignedInt,
    Long,
    UnsignedLong,
    LongLong,
    UnsignedLongLong,
    Float,
    Double,
    LongDouble,
    Bool,
    FloatComplex,
    DoubleComplex,
    LongDoubleComplex,
}

bitflags! {
    struct TypeQualifiers: u8 {
        const CONST    = 1;
        const VOLATILE = 1 << 1;
        const RESTRICT = 1 << 2; // Only for pointers
        const ATOMIC   = 1 << 3;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Enumerator(String, Option<u32>);

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tag {
    Enum(Option<Vec<Enumerator>>),
    // Struct,
    // Union,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum UnqualifiedType {
    Basic(BasicType),
    Tag(Option<String>, Tag),
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QualifiedType(UnqualifiedType, TypeQualifiers);

#[derive(Debug, Clone, PartialEq, Eq)]
enum ArraySize {
    Unspecified,
    Fixed(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DerivedType(QualifiedType, Vec<Deriv>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct FuncParam(Option<String>, Option<DerivedType>);

#[derive(Debug, Clone, PartialEq, Eq)]
enum FuncParams {
    Undef,
    Defined {
        params: Vec<FuncParam>,
        is_variadic: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Deriv {
    Ptr(TypeQualifiers),
    Func(FuncParams),
    Array(ArraySize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Decl(QualifiedType, Vec<(String, Vec<Deriv>)>);

#[derive(Debug, Clone, PartialEq, Eq)]
enum Linkage {
    External,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ExtDecl {
    Decl(Option<Linkage>, Decl),
    TypeDef(Decl),
    // FuncDef(FuncDef),
}

struct TypeManager {
    types_stack: Vec<HashSet<String>>,
}

impl TypeManager {
    fn new() -> TypeManager {
        TypeManager {
            types_stack: vec![HashSet::new()],
        }
    }

    fn add_type_to_current_scope(&mut self, name: String) -> bool {
        self.types_stack.last_mut().unwrap().insert(name)
    }

    fn is_type_name(&self, name: &str) -> bool {
        self.types_stack
            .iter()
            .rev()
            .any(|types| types.contains(name))
    }
}

struct DeclaratorParenLevel {
    ptr_qualifs: Vec<TypeQualifiers>,
    func_params: Option<FuncParams>,
    array_sizes: Vec<ArraySize>,
}

impl DeclaratorParenLevel {
    fn new() -> DeclaratorParenLevel {
        DeclaratorParenLevel {
            ptr_qualifs: Vec::new(),
            func_params: None,
            array_sizes: Vec::new(),
        }
    }

    fn append_to(self, derivs: &mut Vec<Deriv>) {
        for qualifiers in self.ptr_qualifs {
            derivs.push(Deriv::Ptr(qualifiers));
        }
        if let Some(func_params) = self.func_params {
            derivs.push(Deriv::Func(func_params));
        }
        for array_size in self.array_sizes.into_iter().rev() {
            derivs.push(Deriv::Array(array_size));
        }
    }
}

struct Parser<'a> {
    iter: FailablePeekable<TokenIter<'a>>,
    type_manager: TypeManager,
}

impl<'a> Parser<'a> {
    fn from_code(code: &'a str) -> Parser<'a> {
        let iter = TokenIter::from(code);
        Self::from(iter)
    }

    fn from(iter: TokenIter<'a>) -> Parser<'a> {
        Parser {
            iter: iter.peekable(),
            type_manager: TypeManager::new(),
        }
    }

    fn read_type_qualifier(&mut self) -> Result<Option<TypeQualifiers>, ParseError> {
        let qual = if self.iter.advance_if_kw(Keyword::Const)? {
            Some(TypeQualifiers::CONST)
        } else if self.iter.advance_if_kw(Keyword::Volatile)? {
            Some(TypeQualifiers::VOLATILE)
        } else if self.iter.advance_if_kw(Keyword::Restrict)? {
            Some(TypeQualifiers::RESTRICT)
        } else if self.iter.advance_if_kw(Keyword::Atomic)? {
            Some(TypeQualifiers::ATOMIC)
        } else {
            None
        };
        Ok(qual)
    }

    fn read_type_name(&mut self) -> Result<Option<(String, Position)>, ParseError> {
        let type_manager = &self.type_manager;
        if let Some(PositionedToken(Token::Identifier(ident), pos)) =
            self.iter.next_if(|token| match token {
                PositionedToken(Token::Identifier(ident), _) => {
                    type_manager.is_type_name(ident.as_ref())
                }
                _ => false,
            })? {
            Ok(Some((ident, pos)))
        } else {
            Ok(None)
        }
    }

    fn read_enum(&mut self) -> Result<UnqualifiedType, ParseError> {
        let enum_name = self.iter.next_if_any_ident()?.map(|(ident, _)| ident);
        let unqual_type = if self.iter.advance_if_punc(Punctuator::LeftCurlyBracket)? {
            let mut enum_values = Vec::new();
            loop {
                if self.iter.advance_if_punc(Punctuator::RightCurlyBracket)? {
                    break;
                } else if let Some((value_name, _)) = self.iter.next_if_any_ident()? {
                    if self.iter.advance_if_punc(Punctuator::Equal)? {
                        let value = self.read_constant_value()?;
                        enum_values.push(Enumerator(value_name, Some(value)));
                    } else {
                        enum_values.push(Enumerator(value_name, None));
                    }
                    if self.iter.advance_if_punc(Punctuator::RightCurlyBracket)? {
                        break;
                    } else {
                        self.expect_token(Token::Punctuator(Punctuator::Comma))?;
                    }
                } else {
                    panic!("invalid enum decl");
                }
            }
            UnqualifiedType::Tag(enum_name, Tag::Enum(Some(enum_values)))
        } else {
            UnqualifiedType::Tag(enum_name, Tag::Enum(None))
        };
        Ok(unqual_type)
    }

    fn expect_token(&mut self, expected_token: Token) -> Result<(), ParseError> {
        match self.iter.next()? {
            Some(PositionedToken(token, position)) => {
                if token == expected_token {
                    Ok(())
                } else {
                    let message = format!("expecting {:?}, got {:?}", expected_token, token);
                    Err(ParseError::new_with_position(
                        ParseErrorKind::UnexpectedToken(token),
                        message,
                        position,
                    ))
                }
            }
            None => Err(ParseError::new(
                ParseErrorKind::UnexpectedEOF,
                format!(
                    "unfinished construct at end of line, expecting {:?}",
                    expected_token
                ),
            )),
        }
    }

    // Should be called just after having read an opening parenthesis.
    fn read_func_params(&mut self) -> Result<FuncParams, ParseError> {
        if self.iter.advance_if_punc(Punctuator::RightParenthesis)? {
            // foo() means parameters are undefined
            return Ok(FuncParams::Undef);
        }
        let mut params = Vec::new();
        let mut is_first_param = true;
        let mut is_variadic = false;
        loop {
            let (decl_spec, decl_pos, is_typedef) = self.read_decl_spec()?;
            if is_typedef {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidConstruct,
                    "a function parameter cannot contain a typedef".to_string(),
                    decl_pos.expect("a typedef should have a position"),
                ));
            }
            if is_first_param {
                if let Some(QualifiedType(UnqualifiedType::Basic(BasicType::Void), qualifiers)) =
                    decl_spec
                {
                    if qualifiers.is_empty()
                        && self.iter.advance_if_punc(Punctuator::RightParenthesis)?
                    {
                        // foo(void) means no parameter
                        break;
                    }
                }
            }

            let (ident, derivs) = self.read_declarator()?;
            let param = if decl_spec.is_none() && derivs.is_empty() {
                // No type
                if ident.is_none() {
                    panic!("Should have at least an identifier or a type");
                }
                FuncParam(ident, None)
            } else {
                let decl_spec = decl_spec.unwrap_or_else(|| {
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty(),
                    )
                });
                FuncParam(ident, Some(DerivedType(decl_spec, derivs)))
            };
            params.push(param);

            is_first_param = false;
            if self.iter.advance_if_punc(Punctuator::Comma)? {
                if self.iter.advance_if_punc(Punctuator::Ellipsis)? {
                    is_variadic = true;
                    self.expect_token(Token::Punctuator(Punctuator::RightParenthesis))?;
                    break;
                }
            } else {
                self.expect_token(Token::Punctuator(Punctuator::RightParenthesis))?;
                break;
            }
        }

        Ok(FuncParams::Defined {
            params,
            is_variadic,
        })
    }

    // TODO: The return value should be the AST of an expression
    fn read_constant_value(&mut self) -> Result<u32, ParseError> {
        match self.iter.next()? {
            Some(PositionedToken(Token::IntegerLiteral(literal, repr, _), _)) => {
                Ok(u32::from_str_radix(literal.as_ref(), repr.radix()).unwrap())
            }
            Some(PositionedToken(token, position)) => Err(ParseError::new_with_position(
                ParseErrorKind::UnexpectedToken(token),
                "currently only supporting integer literal constant values".to_string(),
                position,
            )),
            None => Err(ParseError::new(
                ParseErrorKind::UnexpectedEOF,
                "end of line when expecting a constant value".to_string(),
            )),
        }
    }

    // Should be called just after having read an opening square bracket.
    fn read_array_size(&mut self) -> Result<ArraySize, ParseError> {
        if self.iter.advance_if_punc(Punctuator::RightSquareBracket)? {
            return Ok(ArraySize::Unspecified);
        }

        let size = self.read_constant_value()?;
        self.expect_token(Token::Punctuator(Punctuator::RightSquareBracket))?;
        Ok(ArraySize::Fixed(size))
    }

    fn read_declarator(&mut self) -> Result<(Option<String>, Vec<Deriv>), ParseError> {
        let mut levels = Vec::new();
        let mut current_level = DeclaratorParenLevel::new();
        let ident = loop {
            if self.iter.advance_if_punc(Punctuator::Star)? {
                let mut qualifiers = TypeQualifiers::empty();
                while let Some(qualifier) = self.read_type_qualifier()? {
                    qualifiers |= qualifier;
                }
                current_level.ptr_qualifs.push(qualifiers);
            } else if self.iter.advance_if_punc(Punctuator::LeftParenthesis)? {
                levels.push(current_level);
                current_level = DeclaratorParenLevel::new();
            } else {
                levels.push(current_level);
                break self.iter.next_if_any_ident()?.map(|(ident, _)| ident);
            }
        };
        for (i, level) in levels.iter_mut().enumerate().rev() {
            if self.iter.advance_if_punc(Punctuator::LeftParenthesis)? {
                level.func_params = Some(self.read_func_params()?);
            }
            while self.iter.advance_if_punc(Punctuator::LeftSquareBracket)? {
                level.array_sizes.push(self.read_array_size()?);
            }
            if i != 0 {
                self.expect_token(Token::Punctuator(Punctuator::RightParenthesis))?;
            }
        }
        let mut derivs = Vec::new();
        for level in levels {
            level.append_to(&mut derivs);
        }
        Ok((ident, derivs))
    }

    fn make_basic_type(
        type_kw_counts: HashMap<Keyword, u32>,
        pos: &Position,
    ) -> Result<BasicType, ParseError> {
        let basic_type = if type_kw_counts.contains_key(&Keyword::Void) {
            if type_kw_counts.len() != 1 {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "void cannot be mixed with other type keywords".to_string(),
                    pos.clone(),
                ));
            } else {
                BasicType::Void
            }
        } else if type_kw_counts.contains_key(&Keyword::Bool) {
            if type_kw_counts.len() != 1 {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "bool cannot be mixed with other type keywords".to_string(),
                    pos.clone(),
                ));
            } else {
                BasicType::Bool
            }
        } else if type_kw_counts.contains_key(&Keyword::Complex) {
            if type_kw_counts.contains_key(&Keyword::Float) {
                if type_kw_counts.len() != 2 {
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidType,
                        "float _Complex cannot be mixed with other type keywords".to_string(),
                        pos.clone(),
                    ));
                } else {
                    BasicType::FloatComplex
                }
            } else if type_kw_counts.contains_key(&Keyword::Double) {
                if type_kw_counts.contains_key(&Keyword::Long) {
                    if type_kw_counts.len() != 3 {
                        return Err(ParseError::new_with_position(
                            ParseErrorKind::InvalidType,
                            "long double _Complex cannot be mixed with other type keywords"
                                .to_string(),
                            pos.clone(),
                        ));
                    } else {
                        BasicType::LongDoubleComplex
                    }
                } else if type_kw_counts.len() != 2 {
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidType,
                        "double _Complex cannot be mixed with type keywords other that long"
                            .to_string(),
                        pos.clone(),
                    ));
                } else {
                    BasicType::DoubleComplex
                }
            } else {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "_Complex must be either float, double, or long double".to_string(),
                    pos.clone(),
                ));
            }
        } else if type_kw_counts.contains_key(&Keyword::Float) {
            if type_kw_counts.len() != 1 {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "float cannot be mixed with type keywords other than _Complex".to_string(),
                    pos.clone(),
                ));
            } else {
                BasicType::Float
            }
        } else if type_kw_counts.contains_key(&Keyword::Double) {
            if type_kw_counts.contains_key(&Keyword::Long) {
                if type_kw_counts.len() != 2 {
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidType,
                        "long double cannot be mixed with type keywords other that _Complex"
                            .to_string(),
                        pos.clone(),
                    ));
                } else {
                    BasicType::LongDouble
                }
            } else if type_kw_counts.len() != 1 {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "double _Complex cannot be mixed with type keywords other that long"
                        .to_string(),
                    pos.clone(),
                ));
            } else {
                BasicType::Double
            }
        } else if type_kw_counts.contains_key(&Keyword::Char) {
            if type_kw_counts.contains_key(&Keyword::Short)
                || type_kw_counts.contains_key(&Keyword::Long)
                || type_kw_counts.contains_key(&Keyword::Int)
            {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "char cannot be mixed with int, short, or long".to_string(),
                    pos.clone(),
                ));
            }
            if type_kw_counts.contains_key(&Keyword::Signed) {
                if type_kw_counts.contains_key(&Keyword::Unsigned) {
                    assert!(type_kw_counts.len() == 3);
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidType,
                        "an char cannot be both signed and unsigned".to_string(),
                        pos.clone(),
                    ));
                } else {
                    assert!(type_kw_counts.len() == 2);
                    BasicType::SignedChar
                }
            } else if type_kw_counts.contains_key(&Keyword::Unsigned) {
                assert!(type_kw_counts.len() == 2);
                BasicType::UnsignedChar
            } else {
                assert!(type_kw_counts.len() == 1);
                BasicType::Char
            }
        } else if type_kw_counts.contains_key(&Keyword::Short) {
            // Note that "short" and "short int" are the same.
            if type_kw_counts.contains_key(&Keyword::Long) {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "short cannot be mixed with long".to_string(),
                    pos.clone(),
                ));
            }
            if type_kw_counts.contains_key(&Keyword::Signed) {
                if type_kw_counts.contains_key(&Keyword::Unsigned) {
                    assert!(type_kw_counts.len() == 3 || type_kw_counts.len() == 4);
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidType,
                        "a short cannot be both signed and unsigned".to_string(),
                        pos.clone(),
                    ));
                } else {
                    assert!(type_kw_counts.len() == 2 || type_kw_counts.len() == 3);
                    BasicType::Short
                }
            } else if type_kw_counts.contains_key(&Keyword::Unsigned) {
                assert!(type_kw_counts.len() == 2 || type_kw_counts.len() == 3);
                BasicType::UnsignedShort
            } else {
                assert!(type_kw_counts.len() == 1);
                BasicType::Short
            }
        } else if type_kw_counts.contains_key(&Keyword::Long) {
            let long_count = type_kw_counts[&Keyword::Long];
            assert!(long_count > 0);
            if long_count > 2 {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "long long long is invalid".to_string(),
                    pos.clone(),
                ));
            }
            // Note that "long" and "long int" are the same.
            if type_kw_counts.contains_key(&Keyword::Signed) {
                if type_kw_counts.contains_key(&Keyword::Unsigned) {
                    assert!(type_kw_counts.len() == 3 || type_kw_counts.len() == 4);
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidType,
                        "a long cannot be both signed and unsigned".to_string(),
                        pos.clone(),
                    ));
                } else {
                    assert!(type_kw_counts.len() == 2 || type_kw_counts.len() == 3);
                    if long_count == 1 {
                        BasicType::Long
                    } else {
                        BasicType::LongLong
                    }
                }
            } else if type_kw_counts.contains_key(&Keyword::Unsigned) {
                assert!(type_kw_counts.len() == 2 || type_kw_counts.len() == 3);
                if long_count == 1 {
                    BasicType::UnsignedLong
                } else {
                    BasicType::UnsignedLongLong
                }
            } else {
                assert!(type_kw_counts.len() == 1);
                if long_count == 1 {
                    BasicType::Long
                } else {
                    BasicType::LongLong
                }
            }
        } else if type_kw_counts.contains_key(&Keyword::Int) {
            if type_kw_counts.contains_key(&Keyword::Signed) {
                if type_kw_counts.contains_key(&Keyword::Unsigned) {
                    assert!(type_kw_counts.len() == 3);
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidType,
                        "an int cannot be both signed and unsigned".to_string(),
                        pos.clone(),
                    ));
                } else {
                    assert!(type_kw_counts.len() == 2);
                    BasicType::SignedInt
                }
            } else if type_kw_counts.contains_key(&Keyword::Unsigned) {
                assert!(type_kw_counts.len() == 2);
                BasicType::UnsignedInt
            } else {
                assert!(type_kw_counts.len() == 1);
                BasicType::Int
            }
        } else if type_kw_counts.contains_key(&Keyword::Signed) {
            if type_kw_counts.contains_key(&Keyword::Unsigned) {
                assert!(type_kw_counts.len() == 2);
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "a value cannot be both signed and unsigned".to_string(),
                    pos.clone(),
                ));
            } else {
                assert!(type_kw_counts.len() == 1);
                BasicType::SignedInt
            }
        } else if type_kw_counts.contains_key(&Keyword::Unsigned) {
            assert!(type_kw_counts.len() == 1);
            BasicType::UnsignedInt
        } else {
            unreachable!()
        };
        Ok(basic_type)
    }

    // TODO: Add linkage support (and check that they don't go with a typedef)
    fn read_decl_spec(
        &mut self,
    ) -> Result<(Option<QualifiedType>, Option<Position>, bool), ParseError> {
        let mut type_kw_counts = HashMap::new();
        let mut decl_pos = None;
        let mut type_name = None;
        let mut type_qual = TypeQualifiers::empty();
        let mut tag = None;
        let mut is_typedef = false;
        loop {
            match self.iter.next_if_any_kw()? {
                Some((kw, pos)) => {
                    match kw {
                        Keyword::Typedef => is_typedef = true,
                        Keyword::Const => type_qual |= TypeQualifiers::CONST,
                        Keyword::Volatile => type_qual |= TypeQualifiers::CONST,
                        Keyword::Restrict => type_qual |= TypeQualifiers::CONST,
                        Keyword::Atomic => type_qual |= TypeQualifiers::CONST,
                        Keyword::Void
                        | Keyword::Int
                        | Keyword::Long
                        | Keyword::Short
                        | Keyword::Char
                        | Keyword::Signed
                        | Keyword::Unsigned
                        | Keyword::Double
                        | Keyword::Float
                        | Keyword::Bool
                        | Keyword::Complex => {
                            type_kw_counts
                                .entry(kw)
                                .and_modify(|count| *count += 1)
                                .or_insert(1);
                        }
                        Keyword::Enum => {
                            if tag.is_some() {
                                return Err(ParseError::new_with_position(
                                    ParseErrorKind::InvalidConstruct,
                                    "a declaration can only have one tag declaration".to_string(),
                                    pos,
                                ));
                            }
                            tag = Some(self.read_enum()?);
                        }
                        kw => {
                            return Err(ParseError::new_with_position(
                                ParseErrorKind::InvalidConstruct,
                                format!("unexpected keyword {:?} in declaration", kw),
                                pos,
                            ))
                        }
                    };
                    decl_pos = Some(pos.clone());
                }
                None => match self.read_type_name()? {
                    Some((ident, pos)) => {
                        decl_pos = Some(pos);
                        type_name = Some(ident);
                    }
                    None => break,
                },
            }
        }
        let unqual_type = if let Some(type_name) = type_name {
            if !type_kw_counts.is_empty() || tag.is_some() {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    format!(
                        "a declaration cannot have both a basic type and a custom type {}",
                        type_name
                    ),
                    decl_pos.expect("if we have a tag, we must have a position"),
                ));
            }
            Some(UnqualifiedType::Custom(type_name))
        } else if let Some(tag) = tag {
            if !type_kw_counts.is_empty() {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "a declaration cannot have both a basic type and a tag type".to_string(),
                    decl_pos.expect("if we have a tag, we must have a position"),
                ));
            }
            Some(tag)
        } else {
            if type_kw_counts.is_empty() {
                if type_qual.is_empty() {
                    None
                } else {
                    Some(UnqualifiedType::Basic(BasicType::Int))
                }
            } else {
                let basic_type =
                    Self::make_basic_type(type_kw_counts, &decl_pos.as_ref().unwrap())?;
                Some(UnqualifiedType::Basic(basic_type))
            }
        };
        Ok((
            unqual_type.map(|unqual_type| QualifiedType(unqual_type, type_qual)),
            decl_pos,
            is_typedef,
        ))
    }
}

impl<'a> FailableIterator for Parser<'a> {
    type Item = ExtDecl;
    type Error = ParseError;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        if self.iter.peek()?.is_none() {
            return Ok(None);
        }
        let (decl_spec, _, is_typedef) = self.read_decl_spec()?;
        let decl_spec = decl_spec.unwrap_or_else(|| {
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Int),
                TypeQualifiers::empty(),
            )
        });

        let mut declarators = Vec::new();
        if !self.iter.advance_if_punc(Punctuator::Semicolon)? {
            loop {
                let (ident, derivs) = self.read_declarator()?;
                let ident = ident.expect("A decl should have an identifier");

                if is_typedef {
                    if !self.type_manager.add_type_to_current_scope(ident.clone()) {
                        panic!(
                            "You should not redefine a type already defined in the current scope"
                        );
                    }
                }
                declarators.push((ident, derivs));

                match self.iter.next()? {
                    Some(PositionedToken(Token::Punctuator(Punctuator::Semicolon), _)) => break,
                    Some(PositionedToken(Token::Punctuator(Punctuator::Comma), _)) => (),
                    Some(PositionedToken(token, position)) => {
                        let message = format!(
                            "got {:?} in declaration where expecting a comma or semicolor",
                            token
                        );
                        return Err(ParseError::new_with_position(
                            ParseErrorKind::UnexpectedToken(token),
                            message,
                            position,
                        ));
                    }
                    None => {
                        return Err(ParseError::new(
                            ParseErrorKind::UnexpectedEOF,
                            "end of file where expecting a comma or semicolor".to_string(),
                        ))
                    }
                }
            }
        }

        let ext_decl = if is_typedef {
            ExtDecl::TypeDef(Decl(decl_spec, declarators))
        } else {
            ExtDecl::Decl(None, Decl(decl_spec, declarators))
        };
        Ok(Some(ext_decl))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_external_declarations(code: &str) -> Vec<ExtDecl> {
        let parser = Parser::from_code(code);
        match parser.collect() {
            Ok(decls) => decls,
            Err(err) => panic!(r#"Unexpected error {:?} for "{:}""#, err, code),
        }
    }

    #[test]
    fn test_simple_declaration() {
        assert_eq!(parse_external_declarations(r#""#), vec![]);
        assert_eq!(
            parse_external_declarations(r#"abcd;"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty()
                    ),
                    vec![("abcd".to_string(), vec![])]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"(abcd);"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty()
                    ),
                    vec![("abcd".to_string(), vec![])]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"int abcd;"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty()
                    ),
                    vec![("abcd".to_string(), vec![])]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"signed long;"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Long),
                        TypeQualifiers::empty()
                    ),
                    vec![]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"unsigned char abcd;"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::UnsignedChar),
                        TypeQualifiers::empty()
                    ),
                    vec![("abcd".to_string(), vec![])]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"*abcd;"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty()
                    ),
                    vec![(
                        "abcd".to_string(),
                        vec![Deriv::Ptr(TypeQualifiers::empty())]
                    )]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"signed long long int * abcd;"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::LongLong),
                        TypeQualifiers::empty()
                    ),
                    vec![(
                        "abcd".to_string(),
                        vec![Deriv::Ptr(TypeQualifiers::empty())]
                    )]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"const short * abcd;"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Short),
                        TypeQualifiers::CONST
                    ),
                    vec![(
                        "abcd".to_string(),
                        vec![Deriv::Ptr(TypeQualifiers::empty())]
                    )]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"short const * abcd;"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Short),
                        TypeQualifiers::CONST
                    ),
                    vec![(
                        "abcd".to_string(),
                        vec![Deriv::Ptr(TypeQualifiers::empty())]
                    )]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"float * const abcd;"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Float),
                        TypeQualifiers::empty()
                    ),
                    vec![("abcd".to_string(), vec![Deriv::Ptr(TypeQualifiers::CONST)])]
                )
            )]
        );
    }

    #[test]
    fn test_function_declaration() {
        assert_eq!(
            parse_external_declarations(r#"foo();"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty()
                    ),
                    vec![("foo".to_string(), vec![Deriv::Func(FuncParams::Undef)])]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"void foo(void);"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Void),
                        TypeQualifiers::empty()
                    ),
                    vec![(
                        "foo".to_string(),
                        vec![Deriv::Func(FuncParams::Defined {
                            params: vec![],
                            is_variadic: false
                        })]
                    )]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"void foo(int a, char b, ...);"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Void),
                        TypeQualifiers::empty()
                    ),
                    vec![(
                        "foo".to_string(),
                        vec![Deriv::Func(FuncParams::Defined {
                            params: vec![
                                FuncParam(
                                    Some("a".to_string()),
                                    Some(DerivedType(
                                        QualifiedType(
                                            UnqualifiedType::Basic(BasicType::Int),
                                            TypeQualifiers::empty()
                                        ),
                                        vec![]
                                    ))
                                ),
                                FuncParam(
                                    Some("b".to_string()),
                                    Some(DerivedType(
                                        QualifiedType(
                                            UnqualifiedType::Basic(BasicType::Char),
                                            TypeQualifiers::empty()
                                        ),
                                        vec![]
                                    ))
                                )
                            ],
                            is_variadic: true
                        })]
                    )]
                )
            )]
        );
    }

    #[test]
    fn test_function_pointer_declaration() {
        assert_eq!(
            parse_external_declarations(r#"int (*foo)();"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty()
                    ),
                    vec![(
                        "foo".to_string(),
                        vec![
                            Deriv::Func(FuncParams::Undef),
                            Deriv::Ptr(TypeQualifiers::empty())
                        ]
                    )]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"int (*(foo))();"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty()
                    ),
                    vec![(
                        "foo".to_string(),
                        vec![
                            Deriv::Func(FuncParams::Undef),
                            Deriv::Ptr(TypeQualifiers::empty())
                        ]
                    )]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"int (*(*bar)())();"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty()
                    ),
                    vec![(
                        "bar".to_string(),
                        vec![
                            Deriv::Func(FuncParams::Undef),
                            Deriv::Ptr(TypeQualifiers::empty()),
                            Deriv::Func(FuncParams::Undef),
                            Deriv::Ptr(TypeQualifiers::empty())
                        ]
                    )]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"int (*foo())();"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty()
                    ),
                    vec![(
                        "foo".to_string(),
                        vec![
                            Deriv::Func(FuncParams::Undef),
                            Deriv::Ptr(TypeQualifiers::empty()),
                            Deriv::Func(FuncParams::Undef)
                        ]
                    )]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"char const * (**hogehoge)();"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Char),
                        TypeQualifiers::CONST
                    ),
                    vec![(
                        "hogehoge".to_string(),
                        vec![
                            Deriv::Ptr(TypeQualifiers::empty()),
                            Deriv::Func(FuncParams::Undef),
                            Deriv::Ptr(TypeQualifiers::empty()),
                            Deriv::Ptr(TypeQualifiers::empty())
                        ]
                    )]
                )
            )]
        );
    }

    #[test]
    fn test_array_declaration() {
        assert_eq!(
            parse_external_declarations(r#"int foo[10];"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty()
                    ),
                    vec![("foo".to_string(), vec![Deriv::Array(ArraySize::Fixed(10))])]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"char foo[1][3];"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Char),
                        TypeQualifiers::empty()
                    ),
                    vec![(
                        "foo".to_string(),
                        vec![
                            Deriv::Array(ArraySize::Fixed(3)),
                            Deriv::Array(ArraySize::Fixed(1))
                        ]
                    )]
                )
            )]
        );

        assert_eq!(
            // Unspecified size arrays should only in function parameters
            parse_external_declarations(r#"void bar(short foo[]);"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Void),
                        TypeQualifiers::empty()
                    ),
                    vec![(
                        "bar".to_string(),
                        vec![Deriv::Func(FuncParams::Defined {
                            params: vec![FuncParam(
                                Some("foo".to_string()),
                                Some(DerivedType(
                                    QualifiedType(
                                        UnqualifiedType::Basic(BasicType::Short),
                                        TypeQualifiers::empty()
                                    ),
                                    vec![Deriv::Array(ArraySize::Unspecified)]
                                ))
                            )],
                            is_variadic: false
                        })]
                    )]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"char *(*abcdef[3])();"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Char),
                        TypeQualifiers::empty()
                    ),
                    vec![(
                        "abcdef".to_string(),
                        vec![
                            Deriv::Ptr(TypeQualifiers::empty()),
                            Deriv::Func(FuncParams::Undef),
                            Deriv::Ptr(TypeQualifiers::empty()),
                            Deriv::Array(ArraySize::Fixed(3))
                        ]
                    )]
                )
            )]
        );
    }

    #[test]
    fn test_simple_type_definition() {
        assert_eq!(
            parse_external_declarations(r#"typedef signed *truc();"#),
            vec![ExtDecl::TypeDef(Decl(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::SignedInt),
                    TypeQualifiers::empty()
                ),
                vec![(
                    "truc".to_string(),
                    vec![
                        Deriv::Ptr(TypeQualifiers::empty()),
                        Deriv::Func(FuncParams::Undef)
                    ]
                )]
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"typedef int *ptr; const ptr foo;"#),
            vec![
                ExtDecl::TypeDef(Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty()
                    ),
                    vec![("ptr".to_string(), vec![Deriv::Ptr(TypeQualifiers::empty())])]
                )),
                ExtDecl::Decl(
                    None,
                    Decl(
                        QualifiedType(
                            UnqualifiedType::Custom("ptr".to_string()),
                            TypeQualifiers::CONST
                        ),
                        vec![("foo".to_string(), vec![])]
                    )
                )
            ]
        );
        assert_eq!(
            parse_external_declarations(r#"typedef int i, *ptr; ptr foo, *bar;"#),
            vec![
                ExtDecl::TypeDef(Decl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty()
                    ),
                    vec![
                        ("i".to_string(), vec![]),
                        ("ptr".to_string(), vec![Deriv::Ptr(TypeQualifiers::empty())])
                    ]
                )),
                ExtDecl::Decl(
                    None,
                    Decl(
                        QualifiedType(
                            UnqualifiedType::Custom("ptr".to_string()),
                            TypeQualifiers::empty()
                        ),
                        vec![
                            ("foo".to_string(), vec![]),
                            ("bar".to_string(), vec![Deriv::Ptr(TypeQualifiers::empty())])
                        ]
                    )
                )
            ]
        );
        assert_eq!(
            // A typed declared and used in one typedef
            parse_external_declarations(r#"typedef int *foo, bar(foo x);"#),
            vec![ExtDecl::TypeDef(Decl(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty()
                ),
                vec![
                    ("foo".to_string(), vec![Deriv::Ptr(TypeQualifiers::empty())]),
                    (
                        "bar".to_string(),
                        vec![Deriv::Func(FuncParams::Defined {
                            params: vec![FuncParam(
                                Some("x".to_string()),
                                Some(DerivedType(
                                    QualifiedType(
                                        UnqualifiedType::Custom("foo".to_string()),
                                        TypeQualifiers::empty()
                                    ),
                                    vec![]
                                ))
                            )],
                            is_variadic: false
                        })]
                    )
                ]
            ))]
        );
        assert_eq!(
            // typedef, basic types keywords can be in any order 😢
            parse_external_declarations(r#"long typedef long unsigned foo;"#),
            vec![ExtDecl::TypeDef(Decl(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::UnsignedLongLong),
                    TypeQualifiers::empty()
                ),
                vec![("foo".to_string(), vec![])]
            ))]
        );
    }

    #[test]
    fn test_tag_definition() {
        assert_eq!(
            parse_external_declarations(r#"enum foo;"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Tag(Some("foo".to_string()), Tag::Enum(None)),
                        TypeQualifiers::empty()
                    ),
                    vec![]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"enum foo bar;"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Tag(Some("foo".to_string()), Tag::Enum(None)),
                        TypeQualifiers::empty()
                    ),
                    vec![("bar".to_string(), vec![])]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"enum foo { a, b = 10, c } bar;"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Tag(
                            Some("foo".to_string()),
                            Tag::Enum(Some(vec![
                                Enumerator("a".to_string(), None),
                                Enumerator("b".to_string(), Some(10)),
                                Enumerator("c".to_string(), None),
                            ]))
                        ),
                        TypeQualifiers::empty()
                    ),
                    vec![("bar".to_string(), vec![])]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"enum foo { a, b = 10, c } bar(void);"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Tag(
                            Some("foo".to_string()),
                            Tag::Enum(Some(vec![
                                Enumerator("a".to_string(), None),
                                Enumerator("b".to_string(), Some(10)),
                                Enumerator("c".to_string(), None),
                            ]))
                        ),
                        TypeQualifiers::empty()
                    ),
                    vec![(
                        "bar".to_string(),
                        vec![Deriv::Func(FuncParams::Defined {
                            params: vec![],
                            is_variadic: false
                        })]
                    )]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"enum { a, b = 10, c } bar(void);"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Tag(
                            None,
                            Tag::Enum(Some(vec![
                                Enumerator("a".to_string(), None),
                                Enumerator("b".to_string(), Some(10)),
                                Enumerator("c".to_string(), None),
                            ]))
                        ),
                        TypeQualifiers::empty()
                    ),
                    vec![(
                        "bar".to_string(),
                        vec![Deriv::Func(FuncParams::Defined {
                            params: vec![],
                            is_variadic: false
                        })]
                    )]
                )
            )]
        );
        assert_eq!(
            // enum in function parameters - note that in that case the enum is only usable inside the function.
            parse_external_declarations(r#"enum foo { a, b = 10, c } bar(enum hoge { x });"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Tag(
                            Some("foo".to_string()),
                            Tag::Enum(Some(vec![
                                Enumerator("a".to_string(), None),
                                Enumerator("b".to_string(), Some(10)),
                                Enumerator("c".to_string(), None),
                            ]))
                        ),
                        TypeQualifiers::empty()
                    ),
                    vec![(
                        "bar".to_string(),
                        vec![Deriv::Func(FuncParams::Defined {
                            params: vec![FuncParam(
                                None,
                                Some(DerivedType(
                                    QualifiedType(
                                        UnqualifiedType::Tag(
                                            Some("hoge".to_string()),
                                            Tag::Enum(Some(vec![Enumerator(
                                                "x".to_string(),
                                                None
                                            )]))
                                        ),
                                        TypeQualifiers::empty()
                                    ),
                                    vec![]
                                ))
                            )],
                            is_variadic: false
                        })]
                    )]
                )
            )]
        );
        assert_eq!(
            parse_external_declarations(r#"enum foo { a, b = 10, c };"#),
            vec![ExtDecl::Decl(
                None,
                Decl(
                    QualifiedType(
                        UnqualifiedType::Tag(
                            Some("foo".to_string()),
                            Tag::Enum(Some(vec![
                                Enumerator("a".to_string(), None),
                                Enumerator("b".to_string(), Some(10)),
                                Enumerator("c".to_string(), None),
                            ]))
                        ),
                        TypeQualifiers::empty()
                    ),
                    vec![]
                )
            )]
        );
    }
}

fn main() -> Result<(), ParseError> {
    let mut parser = Parser::from_code(r#"x;"#);
    let decls = parser.next()?;
    println!("Declaration: {:?}", decls);
    Ok(())
}
