// TODO:
// - For testing pragmas, have a look at clang's test/Sema/pragma-align-packed.c
// - Move tests to one (or multiple) other files
// - Add position to declarations
// - Function definition
// - Variable initialization
// - Error on _Thread_local of types or var decls

mod error;
mod failable;
mod lex;
mod peeking;

use crate::error::{ParseError, ParseErrorKind};
use crate::failable::{FailableIterator, FailablePeekable};
use crate::lex::{Keyword, Position, PositionedToken, Punctuator, Token, TokenIter};

use bitflags::bitflags;
use std::collections::HashSet;

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

bitflags! {
    struct FuncSpecifiers: u8 {
        const INLINE   = 1;
        const NORETURN = 1 << 1;
    }
}

bitflags! {
    struct TypeKeywords: u16 {
        const VOID     = 1;
        const BOOL     = 1 << 1;
        const CHAR     = 1 << 2;
        const SHORT    = 1 << 3;
        const INT      = 1 << 4;
        const LONG_1   = 1 << 5;
        const LONG_2   = 1 << 6;
        const SIGNED   = 1 << 7;
        const UNSIGNED = 1 << 8;
        const DOUBLE   = 1 << 9;
        const FLOAT    = 1 << 10;
        const COMPLEX  = 1 << 11;

        const SIGNED_CHAR = Self::SIGNED.bits | Self::CHAR.bits;
        const UNSIGNED_CHAR = Self::UNSIGNED.bits | Self::CHAR.bits;
        const SHORT_INT = Self::SHORT.bits | Self::INT.bits;
        const SIGNED_SHORT = Self::SIGNED.bits | Self::SHORT.bits;
        const SIGNED_SHORT_INT = Self::SIGNED.bits | Self::SHORT.bits | Self::INT.bits;
        const UNSIGNED_SHORT = Self::UNSIGNED.bits | Self::SHORT.bits;
        const UNSIGNED_SHORT_INT = Self::UNSIGNED.bits | Self::SHORT.bits | Self::INT.bits;
        const SIGNED_INT = Self::SIGNED.bits | Self::INT.bits;
        const UNSIGNED_INT = Self::UNSIGNED.bits | Self::INT.bits;
        const LONG = Self::LONG_1.bits;
        const LONG_INT = Self::LONG_1.bits | Self::INT.bits;
        const SIGNED_LONG = Self::SIGNED.bits | Self::LONG_1.bits;
        const SIGNED_LONG_INT = Self::SIGNED.bits | Self::LONG_1.bits | Self::INT.bits;
        const UNSIGNED_LONG = Self::UNSIGNED.bits | Self::LONG_1.bits;
        const UNSIGNED_LONG_INT = Self::UNSIGNED.bits | Self::LONG_1.bits | Self::INT.bits;
        const LONG_LONG = Self::LONG_1.bits | Self::LONG_2.bits;
        const LONG_LONG_INT = Self::LONG_1.bits | Self::LONG_2.bits | Self::INT.bits;
        const SIGNED_LONG_LONG = Self::SIGNED.bits | Self::LONG_1.bits | Self::LONG_2.bits;
        const SIGNED_LONG_LONG_INT = Self::SIGNED.bits | Self::LONG_1.bits | Self::LONG_2.bits | Self::INT.bits;
        const UNSIGNED_LONG_LONG = Self::UNSIGNED.bits | Self::LONG_1.bits | Self::LONG_2.bits;
        const UNSIGNED_LONG_LONG_INT = Self::UNSIGNED.bits | Self::LONG_1.bits | Self::LONG_2.bits | Self::INT.bits;
        const LONG_DOUBLE = Self::LONG_1.bits | Self::DOUBLE.bits;
        const COMPLEX_FLOAT = Self::COMPLEX.bits | Self::FLOAT.bits;
        const COMPLEX_DOUBLE = Self::COMPLEX.bits | Self::DOUBLE.bits;
        const COMPLEX_LONG_DOUBLE = Self::COMPLEX.bits | Self::LONG_1.bits | Self::DOUBLE.bits;
    }
}

type ConstExpr = u32;

#[derive(Debug, Clone, PartialEq, Eq)]
struct EnumItem(String, Option<ConstExpr>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct TagItemDecl(QualifiedType, Vec<(String, Vec<Deriv>, Option<ConstExpr>)>);

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tag {
    Enum(Option<String>, Option<Vec<EnumItem>>),
    Struct(Option<String>, Option<Vec<TagItemDecl>>),
    Union(Option<String>, Option<Vec<TagItemDecl>>),
}

// impl Tag {
//     fn name(&self) -> &Option<String> {
//         match self {
//             Tag::Enum(name, _) => name,
//             Tag::Struct(name, _) => name,
//             Tag::Union(name, _) => name,
//         }
//     }
// }

#[derive(Debug, Clone, PartialEq, Eq)]
enum UnqualifiedType {
    Basic(BasicType),
    Tag(Tag),
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QualifiedType(UnqualifiedType, TypeQualifiers);

#[derive(Debug, Clone, PartialEq, Eq)]
enum ArraySize {
    Unspecified,
    Fixed(ConstExpr),
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
struct TypeDecl(QualifiedType, Vec<(String, Vec<Deriv>)>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct Decl(
    Option<Linkage>,
    QualifiedType,
    Vec<(String, Vec<Deriv>)>,
    Option<ConstExpr>,
);

#[derive(Debug, Clone, PartialEq, Eq)]
struct FuncDef(
    String,
    Option<Linkage>,
    FuncSpecifiers,
    QualifiedType,
    FuncParams,
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Linkage {
    External,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ExtDecl {
    Decl(Decl),
    TypeDef(TypeDecl),
    FuncDef(FuncDef),
}

#[derive(Debug, Clone)]
enum DeclSpec {
    Decl {
        qual_type: QualifiedType,
        pos: Position,
        linkage: Option<Linkage>,
        func_specifiers: FuncSpecifiers,
        is_thread_local: bool,
    },
    TypeDef {
        qual_type: QualifiedType,
        pos: Position,
    },
}

struct TypeManager {
    types_stack: Vec<HashSet<String>>,
}

impl TypeManager {
    fn builtin_types() -> HashSet<String> {
        let mut builtins = HashSet::new();
        builtins.insert("__builtin_va_list".to_string());
        builtins
    }

    fn new() -> TypeManager {
        TypeManager {
            types_stack: vec![Self::builtin_types()],
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FuncDefKind {
    KnR,
    Ansi,
}

impl FuncDefKind {
    fn from(derivs: &Vec<Deriv>) -> Option<FuncDefKind> {
        if let Some(Deriv::Func(params)) = derivs.first() {
            if let FuncParams::Defined {
                params,
                is_variadic,
            } = params
            {
                if params.len() > 0 && !is_variadic && params.iter().all(|param| {
                    let FuncParam(name, _) = param;
                    name.is_some()
                }) {
                    Some(FuncDefKind::KnR)
                } else {
                    Some(FuncDefKind::Ansi)
                }
            } else {
                Some(FuncDefKind::KnR)
            }
        } else {
            None
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

    fn read_enum(&mut self, pos: &Position) -> Result<Tag, ParseError> {
        let enum_name = self.iter.next_if_any_ident()?.map(|(ident, _)| ident);
        let unqual_type = if self.iter.advance_if_punc(Punctuator::LeftCurlyBracket)? {
            let mut enum_values = Vec::new();
            loop {
                if self.iter.advance_if_punc(Punctuator::RightCurlyBracket)? {
                    break;
                } else if let Some((value_name, _)) = self.iter.next_if_any_ident()? {
                    if self.iter.advance_if_punc(Punctuator::Equal)? {
                        let value = self.read_constant_value()?;
                        enum_values.push(EnumItem(value_name, Some(value)));
                    } else {
                        enum_values.push(EnumItem(value_name, None));
                    }
                    if self.iter.advance_if_punc(Punctuator::RightCurlyBracket)? {
                        break;
                    } else {
                        self.expect_token(&Token::Punctuator(Punctuator::Comma))?;
                    }
                } else {
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidConstruct,
                        "invalid enum declaration".to_string(),
                        pos.clone(),
                    ));
                }
            }
            Tag::Enum(enum_name, Some(enum_values))
        } else {
            if enum_name.is_none() {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidConstruct,
                    "an enum declaration must have either a name or the list of values".to_string(),
                    pos.clone(),
                ));
            }
            Tag::Enum(enum_name, None)
        };
        Ok(unqual_type)
    }

    fn read_union_or_struct<F>(&mut self, pos: &Position, builder: F) -> Result<Tag, ParseError>
    where
        F: Fn(Option<String>, Option<Vec<TagItemDecl>>) -> Tag,
    {
        let struct_name = self.iter.next_if_any_ident()?.map(|(ident, _)| ident);
        let tag = if self.iter.advance_if_punc(Punctuator::LeftCurlyBracket)? {
            let mut struct_field_decls = Vec::new();
            loop {
                if self.iter.advance_if_punc(Punctuator::RightCurlyBracket)? {
                    break;
                }
                let root_type = match self.read_decl_spec()? {
                    Some(DeclSpec::TypeDef { pos, .. }) => {
                        return Err(ParseError::new_with_position(
                            ParseErrorKind::InvalidConstruct,
                            "a struct field cannot contain a typedef".to_string(),
                            pos,
                        ));
                    }
                    Some(DeclSpec::Decl {
                        qual_type,
                        pos,
                        linkage,
                        ..
                    }) => {
                        if linkage.is_some() {
                            return Err(ParseError::new_with_position(
                                ParseErrorKind::InvalidConstruct,
                                "a struct field cannot be extern or static".to_string(),
                                pos,
                            ));
                        }
                        qual_type
                    }
                    None => {
                        return Err(ParseError::new_with_position(
                            ParseErrorKind::InvalidConstruct,
                            "a struct field requires a type".to_string(),
                            pos.clone(),
                        ))
                    }
                };
                let mut fields = Vec::new();
                let mut did_finish_struct = false;
                loop {
                    if self.iter.advance_if_punc(Punctuator::RightCurlyBracket)? {
                        did_finish_struct = true;
                        break;
                    } else if self.iter.advance_if_punc(Punctuator::Semicolon)? {
                        break;
                    }

                    let (ident, derivs) = self.read_declarator()?;
                    let ident = if let Some(ident) = ident {
                        ident
                    } else {
                        return Err(ParseError::new_with_position(
                            ParseErrorKind::InvalidConstruct,
                            "a struct field requires a name in most cases".to_string(),
                            pos.clone(),
                        ));
                    };

                    let bit_size = if self.iter.advance_if_punc(Punctuator::Colon)? {
                        Some(self.read_constant_value()?)
                    } else {
                        None
                    };

                    fields.push((ident, derivs, bit_size));

                    if self.iter.advance_if_punc(Punctuator::RightCurlyBracket)? {
                        did_finish_struct = true;
                        break;
                    } else if self.iter.advance_if_punc(Punctuator::Semicolon)? {
                        break;
                    } else {
                        self.expect_token(&Token::Punctuator(Punctuator::Comma))?;
                    }
                }
                struct_field_decls.push(TagItemDecl(root_type, fields));
                if did_finish_struct {
                    break;
                }
            }
            builder(struct_name, Some(struct_field_decls))
        } else {
            if struct_name.is_none() {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidConstruct,
                    "a struct declaration must have either a name or the list of fields"
                        .to_string(),
                    pos.clone(),
                ));
            }
            builder(struct_name, None)
        };
        Ok(tag)
    }

    fn expect_token(&mut self, expected: &Token) -> Result<Position, ParseError> {
        match self.iter.next()? {
            Some(PositionedToken(token, pos)) => {
                if token == *expected {
                    Ok(pos)
                } else {
                    let message = format!("expecting {:?}, got {:?}", expected, token);
                    Err(ParseError::new_with_position(
                        ParseErrorKind::UnexpectedToken(token),
                        message,
                        pos,
                    ))
                }
            }
            None => Err(ParseError::new(
                ParseErrorKind::UnexpectedEOF,
                format!(
                    "unfinished construct at end of line, expecting {:?}",
                    expected
                ),
            )),
        }
    }

    fn next_token_pos(&mut self) -> Result<Option<Position>, ParseError> {
        match self.iter.peek()? {
            Some(PositionedToken(_, pos)) => Ok(Some(pos.clone())),
            None => Ok(None),
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
            let root_type = match self.read_decl_spec()? {
                Some(DeclSpec::TypeDef { pos, .. }) => {
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidConstruct,
                        "a function parameter cannot contain a typedef".to_string(),
                        pos,
                    ));
                }
                Some(DeclSpec::Decl {
                    qual_type,
                    pos,
                    linkage,
                    ..
                }) => {
                    if linkage.is_some() {
                        return Err(ParseError::new_with_position(
                            ParseErrorKind::InvalidConstruct,
                            "a function parameter cannot be extern or static".to_string(),
                            pos,
                        ));
                    }
                    Some(qual_type)
                }
                None => None,
            };
            if is_first_param {
                if let Some(QualifiedType(UnqualifiedType::Basic(BasicType::Void), qualifiers)) =
                    root_type
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
            let param = if root_type.is_none() && derivs.is_empty() {
                // No type
                if ident.is_none() {
                    match self.next_token_pos()? {
                        Some(pos) => {
                            return Err(ParseError::new_with_position(
                                ParseErrorKind::InvalidConstruct,
                                "a declaration should have at least an identifier or a type"
                                    .to_string(),
                                pos.clone(),
                            ))
                        }
                        None => {
                            return Err(ParseError::new(
                                ParseErrorKind::UnexpectedEOF,
                                "unfinished declaration at the end of the file".to_string(),
                            ))
                        }
                    }
                }
                FuncParam(ident, None)
            } else {
                let root_type = root_type.unwrap_or_else(|| {
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty(),
                    )
                });
                FuncParam(ident, Some(DerivedType(root_type, derivs)))
            };
            params.push(param);

            is_first_param = false;
            if self.iter.advance_if_punc(Punctuator::Comma)? {
                if self.iter.advance_if_punc(Punctuator::Ellipsis)? {
                    is_variadic = true;
                    self.expect_token(&Token::Punctuator(Punctuator::RightParenthesis))?;
                    break;
                }
            } else {
                self.expect_token(&Token::Punctuator(Punctuator::RightParenthesis))?;
                break;
            }
        }

        Ok(FuncParams::Defined {
            params,
            is_variadic,
        })
    }

    // TODO: The return value should be the AST of an expression
    fn read_constant_value(&mut self) -> Result<ConstExpr, ParseError> {
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
        self.expect_token(&Token::Punctuator(Punctuator::RightSquareBracket))?;
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
                self.expect_token(&Token::Punctuator(Punctuator::RightParenthesis))?;
            }
        }
        let mut derivs = Vec::new();
        for level in levels {
            level.append_to(&mut derivs);
        }
        Ok((ident, derivs))
    }

    fn make_basic_type(type_kws: TypeKeywords, pos: &Position) -> Result<BasicType, ParseError> {
        let basic_type = match type_kws {
            TypeKeywords::VOID => BasicType::Void,
            TypeKeywords::BOOL => BasicType::Bool,
            TypeKeywords::CHAR => BasicType::Char,
            TypeKeywords::SIGNED_CHAR => BasicType::SignedChar,
            TypeKeywords::UNSIGNED_CHAR => BasicType::UnsignedChar,
            TypeKeywords::SHORT
            | TypeKeywords::SHORT_INT
            | TypeKeywords::SIGNED_SHORT
            | TypeKeywords::SIGNED_SHORT_INT => BasicType::Short,
            TypeKeywords::UNSIGNED_SHORT | TypeKeywords::UNSIGNED_SHORT_INT => {
                BasicType::UnsignedShort
            }
            TypeKeywords::INT => BasicType::Int,
            TypeKeywords::SIGNED | TypeKeywords::SIGNED_INT => BasicType::SignedInt,
            TypeKeywords::UNSIGNED | TypeKeywords::UNSIGNED_INT => BasicType::UnsignedInt,
            TypeKeywords::LONG
            | TypeKeywords::LONG_INT
            | TypeKeywords::SIGNED_LONG
            | TypeKeywords::SIGNED_LONG_INT => BasicType::Long,
            TypeKeywords::UNSIGNED_LONG | TypeKeywords::UNSIGNED_LONG_INT => {
                BasicType::UnsignedLong
            }
            TypeKeywords::LONG_LONG
            | TypeKeywords::LONG_LONG_INT
            | TypeKeywords::SIGNED_LONG_LONG
            | TypeKeywords::SIGNED_LONG_LONG_INT => BasicType::LongLong,
            TypeKeywords::UNSIGNED_LONG_LONG | TypeKeywords::UNSIGNED_LONG_LONG_INT => {
                BasicType::UnsignedLongLong
            }
            TypeKeywords::DOUBLE => BasicType::Double,
            TypeKeywords::FLOAT => BasicType::Float,
            TypeKeywords::COMPLEX => BasicType::DoubleComplex,
            TypeKeywords::LONG_DOUBLE => BasicType::LongDouble,
            TypeKeywords::COMPLEX_FLOAT => BasicType::FloatComplex,
            TypeKeywords::COMPLEX_DOUBLE => BasicType::DoubleComplex,
            TypeKeywords::COMPLEX_LONG_DOUBLE => BasicType::LongDoubleComplex,
            _ => {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "invalid combination of basic type keywords".to_string(),
                    pos.clone(),
                ))
            }
        };
        Ok(basic_type)
    }

    fn read_decl_spec(&mut self) -> Result<Option<DeclSpec>, ParseError> {
        let mut type_kws = TypeKeywords::empty();
        let mut decl_pos = None;
        let mut type_name = None;
        let mut type_qual = TypeQualifiers::empty();
        let mut tag = None;
        let mut is_typedef = false;
        let mut has_useless_kw = false;
        let mut linkage = None;
        let mut is_thread_local = false;
        let mut func_specifiers = FuncSpecifiers::empty();
        loop {
            match self.iter.next_if_any_kw()? {
                Some((kw, pos)) => {
                    match kw {
                        Keyword::Typedef => is_typedef = true,
                        Keyword::Const => type_qual |= TypeQualifiers::CONST,
                        Keyword::Volatile => type_qual |= TypeQualifiers::CONST,
                        Keyword::Restrict => type_qual |= TypeQualifiers::CONST,
                        Keyword::Atomic => type_qual |= TypeQualifiers::CONST,
                        // auto and register don't mean anything anymore
                        Keyword::Auto | Keyword::Register => has_useless_kw = true,
                        Keyword::Void => type_kws |= TypeKeywords::VOID,
                        Keyword::Int => type_kws |= TypeKeywords::INT,
                        Keyword::Long => {
                            // When an expression has multiple "long", the first one is LONG_1, and the other ones are LONG_2
                            if type_kws.contains(TypeKeywords::LONG_1) {
                                type_kws |= TypeKeywords::LONG_2
                            } else {
                                type_kws |= TypeKeywords::LONG_1
                            }
                        }
                        Keyword::Short => type_kws |= TypeKeywords::SHORT,
                        Keyword::Char => type_kws |= TypeKeywords::CHAR,
                        Keyword::Signed => type_kws |= TypeKeywords::SIGNED,
                        Keyword::Unsigned => type_kws |= TypeKeywords::UNSIGNED,
                        Keyword::Double => type_kws |= TypeKeywords::DOUBLE,
                        Keyword::Float => type_kws |= TypeKeywords::FLOAT,
                        Keyword::Bool => type_kws |= TypeKeywords::BOOL,
                        Keyword::Complex => type_kws |= TypeKeywords::COMPLEX,
                        Keyword::Enum | Keyword::Struct | Keyword::Union => {
                            if tag.is_some() {
                                return Err(ParseError::new_with_position(
                                    ParseErrorKind::InvalidConstruct,
                                    "a declaration can only have one tag declaration".to_string(),
                                    pos,
                                ));
                            }
                            tag = Some(match kw {
                                Keyword::Enum => self.read_enum(&pos)?,
                                Keyword::Struct => self.read_union_or_struct(&pos, Tag::Struct)?,
                                Keyword::Union => self.read_union_or_struct(&pos, Tag::Union)?,
                                _ => unreachable!(),
                            });
                        }
                        Keyword::Extern => {
                            if linkage == Some(Linkage::Internal) {
                                return Err(ParseError::new_with_position(
                                    ParseErrorKind::InvalidConstruct,
                                    "a declaration cannot be both extern and static".to_string(),
                                    pos,
                                ));
                            }
                            linkage = Some(Linkage::External);
                        }
                        Keyword::Static => {
                            if linkage == Some(Linkage::External) {
                                return Err(ParseError::new_with_position(
                                    ParseErrorKind::InvalidConstruct,
                                    "a declaration cannot be both extern and static".to_string(),
                                    pos,
                                ));
                            }
                            linkage = Some(Linkage::Internal);
                        }
                        Keyword::Noreturn => func_specifiers |= FuncSpecifiers::NORETURN,
                        Keyword::Inline => func_specifiers |= FuncSpecifiers::INLINE,
                        Keyword::ThreadLocal => is_thread_local = true,
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
        let decl_pos = match decl_pos {
            Some(decl_pos) => decl_pos,
            None => {
                assert!(
                    !is_typedef
                        && type_kws.is_empty()
                        && linkage.is_none()
                        && tag.is_none()
                        && !has_useless_kw
                );
                return Ok(None);
            }
        };
        let unqual_type = if let Some(type_name) = type_name {
            if !type_kws.is_empty() || tag.is_some() {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    format!(
                        "a declaration cannot have both a basic type and a custom type {}",
                        type_name
                    ),
                    decl_pos,
                ));
            }
            UnqualifiedType::Custom(type_name)
        } else if let Some(tag) = tag {
            if !type_kws.is_empty() {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "a declaration cannot have both a basic type and a tag type".to_string(),
                    decl_pos,
                ));
            }
            UnqualifiedType::Tag(tag)
        } else if type_kws.is_empty() {
            UnqualifiedType::Basic(BasicType::Int)
        } else {
            let basic_type = Self::make_basic_type(type_kws, &decl_pos)?;
            UnqualifiedType::Basic(basic_type)
        };
        let qual_type = QualifiedType(unqual_type, type_qual);
        let decl_spec = if is_typedef {
            if linkage.is_some() {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "a typedef cannot use static or extern".to_string(),
                    decl_pos,
                ));
            }
            if !func_specifiers.is_empty() {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "a typedef cannot use inline or _Noreturn".to_string(),
                    decl_pos,
                ));
            }
            if is_thread_local {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "a typedef cannot use _Thread_local".to_string(),
                    decl_pos,
                ));
            }
            DeclSpec::TypeDef {
                qual_type,
                pos: decl_pos,
            }
        } else {
            DeclSpec::Decl {
                qual_type,
                linkage,
                pos: decl_pos,
                func_specifiers,
                is_thread_local,
            }
        };
        Ok(Some(decl_spec))
    }

    fn read_func_def(
        &mut self,
        func_def_kind: FuncDefKind,
        ident: String,
        decl_spec: Option<DeclSpec>,
        mut derivs: Vec<Deriv>,
    ) -> Result<ExtDecl, ParseError> {
        let pos = self.expect_token(&Token::Punctuator(Punctuator::LeftCurlyBracket))?;

        let params = if let Deriv::Func(params) = derivs.remove(0) {
            params
        } else {
            unreachable!()
        };

        let mut opened_curlies = 1;
        loop {
            match self.iter.next()? {
                Some(PositionedToken(Token::Punctuator(Punctuator::LeftCurlyBracket), _)) => {
                    opened_curlies += 1
                }
                Some(PositionedToken(Token::Punctuator(Punctuator::RightCurlyBracket), _)) => {
                    opened_curlies -= 1;
                    if opened_curlies == 0 {
                        break;
                    }
                }
                None => {
                    return Err(ParseError::new(
                        ParseErrorKind::UnexpectedEOF,
                        format!(
                            "end of file even though {} curly braces were not closed",
                            opened_curlies
                        ),
                    ))
                }
                _ => (),
            }
        }

        let (qual_type, linkage, func_specifiers, is_thread_local) = match decl_spec {
            Some(DeclSpec::Decl {
                qual_type,
                linkage,
                func_specifiers,
                is_thread_local,
                ..
            }) => (qual_type, linkage, func_specifiers, is_thread_local),
            None => (
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty(),
                ),
                None,
                FuncSpecifiers::empty(),
                false,
            ),
            _ => unreachable!(),
        };

        if is_thread_local {
            return Err(ParseError::new_with_position(
                ParseErrorKind::InvalidConstruct,
                "only variable declaration can be thread local".to_string(),
                pos,
            ));
        }

        let def = FuncDef(ident, linkage, func_specifiers, qual_type, params);
        return Ok(ExtDecl::FuncDef(def));
    }
}

impl<'a> FailableIterator for Parser<'a> {
    type Item = ExtDecl;
    type Error = ParseError;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        if self.iter.peek()?.is_none() {
            return Ok(None);
        }
        let decl_spec = self.read_decl_spec()?;
        let is_typedef = match &decl_spec {
            Some(DeclSpec::TypeDef { .. }) => true,
            _ => false,
        };

        let mut declarators = Vec::new();
        if !self.iter.advance_if_punc(Punctuator::Semicolon)? {
            loop {
                let (ident, derivs) = self.read_declarator()?;
                let ident = match ident {
                    Some(ident) => ident,
                    None => {
                        return Err(match self.next_token_pos()? {
                            Some(pos) => ParseError::new_with_position(
                                ParseErrorKind::InvalidConstruct,
                                "a declaration should have an identifier".to_string(),
                                pos,
                            ),
                            None => ParseError::new(
                                ParseErrorKind::UnexpectedEOF,
                                "unfinished declaration at the end of the file".to_string(),
                            ),
                        })
                    }
                };

                if is_typedef && !self.type_manager.add_type_to_current_scope(ident.clone()) {
                    let pos = if let Some(DeclSpec::TypeDef { pos, .. }) = decl_spec {
                        pos
                    } else {
                        unreachable!();
                    };
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidConstruct,
                        format!("trying to redefine the already defined type {}", ident),
                        pos,
                    ));
                }

                let func_def_kind = if !is_typedef && declarators.is_empty() {
                    if let Some(func_def_kind) = FuncDefKind::from(&derivs) {
                        match self.iter.peek()? {
                            // If the next token is a ";" or ",", we're in a function declaration, not definition
                            Some(PositionedToken(Token::Punctuator(Punctuator::Semicolon), _))
                            | Some(PositionedToken(Token::Punctuator(Punctuator::Comma), _)) => {
                                None
                            }
                            _ => Some(func_def_kind),
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(func_def_kind) = func_def_kind {
                    return Ok(Some(self.read_func_def(
                        func_def_kind,
                        ident,
                        decl_spec,
                        derivs,
                    )?));
                }

                declarators.push((ident, derivs));

                match self.iter.next()? {
                    Some(PositionedToken(Token::Punctuator(Punctuator::Semicolon), _)) => break,
                    Some(PositionedToken(Token::Punctuator(Punctuator::Comma), _)) => (),
                    Some(PositionedToken(token, pos)) => {
                        let message = format!(
                            "got {:?} in declaration where expecting a comma or semicolon",
                            token
                        );
                        return Err(ParseError::new_with_position(
                            ParseErrorKind::UnexpectedToken(token),
                            message,
                            pos,
                        ));
                    }
                    None => {
                        return Err(ParseError::new(
                            ParseErrorKind::UnexpectedEOF,
                            "end of file where expecting a comma or semicolon".to_string(),
                        ))
                    }
                }
            }
        }

        let ext_decl = match decl_spec {
            Some(DeclSpec::TypeDef { qual_type, .. }) => {
                ExtDecl::TypeDef(TypeDecl(qual_type, declarators))
            }
            Some(DeclSpec::Decl {
                qual_type, linkage, ..
            }) => ExtDecl::Decl(Decl(linkage, qual_type, declarators, None)),
            None => ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty(),
                ),
                declarators,
                None,
            )),
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
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty()
                ),
                vec![("abcd".to_string(), vec![])],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"(abcd);"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty()
                ),
                vec![("abcd".to_string(), vec![])],
                None,
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"int abcd;"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty()
                ),
                vec![("abcd".to_string(), vec![])],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"signed long;"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Long),
                    TypeQualifiers::empty()
                ),
                vec![],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"unsigned char abcd;"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::UnsignedChar),
                    TypeQualifiers::empty()
                ),
                vec![("abcd".to_string(), vec![])],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"*abcd;"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty()
                ),
                vec![(
                    "abcd".to_string(),
                    vec![Deriv::Ptr(TypeQualifiers::empty())]
                )],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"signed long long int * abcd;"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::LongLong),
                    TypeQualifiers::empty()
                ),
                vec![(
                    "abcd".to_string(),
                    vec![Deriv::Ptr(TypeQualifiers::empty())]
                )],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"const short * abcd;"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Short),
                    TypeQualifiers::CONST
                ),
                vec![(
                    "abcd".to_string(),
                    vec![Deriv::Ptr(TypeQualifiers::empty())]
                )],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"short const * abcd;"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Short),
                    TypeQualifiers::CONST
                ),
                vec![(
                    "abcd".to_string(),
                    vec![Deriv::Ptr(TypeQualifiers::empty())]
                )],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"float * const abcd;"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Float),
                    TypeQualifiers::empty()
                ),
                vec![("abcd".to_string(), vec![Deriv::Ptr(TypeQualifiers::CONST)])],
                None
            ))]
        );
    }

    #[test]
    fn test_function_declaration() {
        assert_eq!(
            parse_external_declarations(r#"foo();"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty()
                ),
                vec![("foo".to_string(), vec![Deriv::Func(FuncParams::Undef)])],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"void foo(void);"#),
            vec![ExtDecl::Decl(Decl(
                None,
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
                )],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"void foo(int a, char b, ...);"#),
            vec![ExtDecl::Decl(Decl(
                None,
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
                )],
                None
            ))]
        );
        // function returning a function pointer
        assert_eq!(
            parse_external_declarations(r#"short (*foo())();"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Short),
                    TypeQualifiers::empty()
                ),
                vec![(
                    "foo".to_string(),
                    vec![
                        Deriv::Func(FuncParams::Undef),
                        Deriv::Ptr(TypeQualifiers::empty()),
                        Deriv::Func(FuncParams::Undef)
                    ]
                )],
                None
            ))]
        );
    }

    #[test]
    fn test_function_definition() {
        assert_eq!(
            parse_external_declarations(r#"foo() {}"#),
            vec![ExtDecl::FuncDef(FuncDef(
                "foo".to_string(),
                None,
                FuncSpecifiers::empty(),
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty()
                ),
                FuncParams::Undef
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"void foo() {}"#),
            vec![ExtDecl::FuncDef(FuncDef(
                "foo".to_string(),
                None,
                FuncSpecifiers::empty(),
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Void),
                    TypeQualifiers::empty()
                ),
                FuncParams::Undef
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"void foo(int a, char b) {}"#),
            vec![ExtDecl::FuncDef(FuncDef(
                "foo".to_string(),
                None,
                FuncSpecifiers::empty(),
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Void),
                    TypeQualifiers::empty()
                ),
                FuncParams::Defined {
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
                    is_variadic: false
                }
            ))]
        );
        // void foo(a) {}
        // void foo(a, b) int a {}
        // short (*foo())() {}
    }

    #[test]
    fn test_function_pointer_declaration() {
        assert_eq!(
            parse_external_declarations(r#"int (*foo)();"#),
            vec![ExtDecl::Decl(Decl(
                None,
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
                )],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"int (*(foo))();"#),
            vec![ExtDecl::Decl(Decl(
                None,
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
                )],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"int (*(*bar)())();"#),
            vec![ExtDecl::Decl(Decl(
                None,
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
                )],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"int (*foo())();"#),
            vec![ExtDecl::Decl(Decl(
                None,
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
                )],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"char const * (**hogehoge)();"#),
            vec![ExtDecl::Decl(Decl(
                None,
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
                )],
                None
            ))]
        );
    }

    #[test]
    fn test_array_declaration() {
        assert_eq!(
            parse_external_declarations(r#"int foo[10];"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty()
                ),
                vec![("foo".to_string(), vec![Deriv::Array(ArraySize::Fixed(10))])],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"char foo[1][3];"#),
            vec![ExtDecl::Decl(Decl(
                None,
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
                )],
                None
            ))]
        );

        assert_eq!(
            // Unspecified size arrays should only in function parameters
            parse_external_declarations(r#"void bar(short foo[]);"#),
            vec![ExtDecl::Decl(Decl(
                None,
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
                )],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"char *(*abcdef[3])();"#),
            vec![ExtDecl::Decl(Decl(
                None,
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
                )],
                None
            ))]
        );
    }

    #[test]
    fn test_simple_type_definition() {
        assert_eq!(
            parse_external_declarations(r#"typedef signed *truc();"#),
            vec![ExtDecl::TypeDef(TypeDecl(
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
                ExtDecl::TypeDef(TypeDecl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty()
                    ),
                    vec![("ptr".to_string(), vec![Deriv::Ptr(TypeQualifiers::empty())])]
                )),
                ExtDecl::Decl(Decl(
                    None,
                    QualifiedType(
                        UnqualifiedType::Custom("ptr".to_string()),
                        TypeQualifiers::CONST
                    ),
                    vec![("foo".to_string(), vec![])],
                    None
                ))
            ]
        );
        assert_eq!(
            parse_external_declarations(r#"typedef int i, *ptr; ptr foo, *bar;"#),
            vec![
                ExtDecl::TypeDef(TypeDecl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty()
                    ),
                    vec![
                        ("i".to_string(), vec![]),
                        ("ptr".to_string(), vec![Deriv::Ptr(TypeQualifiers::empty())])
                    ]
                )),
                ExtDecl::Decl(Decl(
                    None,
                    QualifiedType(
                        UnqualifiedType::Custom("ptr".to_string()),
                        TypeQualifiers::empty()
                    ),
                    vec![
                        ("foo".to_string(), vec![]),
                        ("bar".to_string(), vec![Deriv::Ptr(TypeQualifiers::empty())])
                    ],
                    None
                ))
            ]
        );
        assert_eq!(
            // A typed declared and used in one typedef
            parse_external_declarations(r#"typedef int *foo, bar(foo x);"#),
            vec![ExtDecl::TypeDef(TypeDecl(
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
            vec![ExtDecl::TypeDef(TypeDecl(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::UnsignedLongLong),
                    TypeQualifiers::empty()
                ),
                vec![("foo".to_string(), vec![])]
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"typedef long int __darwin_ptrdiff_t;"#),
            vec![ExtDecl::TypeDef(TypeDecl(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Long),
                    TypeQualifiers::empty()
                ),
                vec![("__darwin_ptrdiff_t".to_string(), vec![])]
            ))]
        );
    }

    #[test]
    fn test_tag_definition() {
        assert_eq!(
            parse_external_declarations(r#"enum foo;"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Tag(Tag::Enum(Some("foo".to_string()), None)),
                    TypeQualifiers::empty()
                ),
                vec![],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"enum foo bar;"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Tag(Tag::Enum(Some("foo".to_string()), None)),
                    TypeQualifiers::empty()
                ),
                vec![("bar".to_string(), vec![])],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"enum foo { a, b = 10, c } bar;"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Tag(Tag::Enum(
                        Some("foo".to_string()),
                        Some(vec![
                            EnumItem("a".to_string(), None),
                            EnumItem("b".to_string(), Some(10)),
                            EnumItem("c".to_string(), None),
                        ])
                    )),
                    TypeQualifiers::empty()
                ),
                vec![("bar".to_string(), vec![])],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"enum foo { a, b = 10, c } bar(void);"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Tag(Tag::Enum(
                        Some("foo".to_string()),
                        Some(vec![
                            EnumItem("a".to_string(), None),
                            EnumItem("b".to_string(), Some(10)),
                            EnumItem("c".to_string(), None),
                        ])
                    )),
                    TypeQualifiers::empty()
                ),
                vec![(
                    "bar".to_string(),
                    vec![Deriv::Func(FuncParams::Defined {
                        params: vec![],
                        is_variadic: false
                    })]
                )],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"enum { a, b = 10, c } bar(void);"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Tag(Tag::Enum(
                        None,
                        Some(vec![
                            EnumItem("a".to_string(), None),
                            EnumItem("b".to_string(), Some(10)),
                            EnumItem("c".to_string(), None),
                        ])
                    )),
                    TypeQualifiers::empty()
                ),
                vec![(
                    "bar".to_string(),
                    vec![Deriv::Func(FuncParams::Defined {
                        params: vec![],
                        is_variadic: false
                    })]
                )],
                None
            ))]
        );
        assert_eq!(
            // enum in function parameters - note that in that case the enum is only usable inside the function.
            parse_external_declarations(r#"enum foo { a, b = 10, c } bar(enum hoge { x });"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Tag(Tag::Enum(
                        Some("foo".to_string()),
                        Some(vec![
                            EnumItem("a".to_string(), None),
                            EnumItem("b".to_string(), Some(10)),
                            EnumItem("c".to_string(), None),
                        ])
                    )),
                    TypeQualifiers::empty()
                ),
                vec![(
                    "bar".to_string(),
                    vec![Deriv::Func(FuncParams::Defined {
                        params: vec![FuncParam(
                            None,
                            Some(DerivedType(
                                QualifiedType(
                                    UnqualifiedType::Tag(Tag::Enum(
                                        Some("hoge".to_string()),
                                        Some(vec![EnumItem("x".to_string(), None)])
                                    )),
                                    TypeQualifiers::empty()
                                ),
                                vec![]
                            ))
                        )],
                        is_variadic: false
                    })]
                )],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"enum foo { a, b = 10, c };"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Tag(Tag::Enum(
                        Some("foo".to_string()),
                        Some(vec![
                            EnumItem("a".to_string(), None),
                            EnumItem("b".to_string(), Some(10)),
                            EnumItem("c".to_string(), None),
                        ])
                    )),
                    TypeQualifiers::empty()
                ),
                vec![],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(r#"struct foo { int a : 1, b; };"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Tag(Tag::Struct(
                        Some("foo".to_string()),
                        Some(vec![TagItemDecl(
                            QualifiedType(
                                UnqualifiedType::Basic(BasicType::Int),
                                TypeQualifiers::empty()
                            ),
                            vec![
                                ("a".to_string(), vec![], Some(1)),
                                ("b".to_string(), vec![], None),
                            ]
                        )])
                    )),
                    TypeQualifiers::empty()
                ),
                vec![],
                None
            ))]
        );
        // self-referencing struct
        assert_eq!(
            parse_external_declarations(r#"struct s { struct s *next };"#),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Tag(Tag::Struct(
                        Some("s".to_string()),
                        Some(vec![TagItemDecl(
                            QualifiedType(
                                UnqualifiedType::Tag(Tag::Struct(Some("s".to_string()), None)),
                                TypeQualifiers::empty()
                            ),
                            vec![(
                                "next".to_string(),
                                vec![Deriv::Ptr(TypeQualifiers::empty())],
                                None
                            )]
                        )])
                    )),
                    TypeQualifiers::empty()
                ),
                vec![],
                None
            ))]
        );
        assert_eq!(
            parse_external_declarations(
                r#"union { int x : 2; struct { int a, *b }; char * const z } foo;"#
            ),
            vec![ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Tag(Tag::Union(
                        None,
                        Some(vec![
                            TagItemDecl(
                                QualifiedType(
                                    UnqualifiedType::Basic(BasicType::Int),
                                    TypeQualifiers::empty()
                                ),
                                vec![("x".to_string(), vec![], Some(2)),]
                            ),
                            TagItemDecl(
                                QualifiedType(
                                    UnqualifiedType::Tag(Tag::Struct(
                                        None,
                                        Some(vec![TagItemDecl(
                                            QualifiedType(
                                                UnqualifiedType::Basic(BasicType::Int),
                                                TypeQualifiers::empty()
                                            ),
                                            vec![
                                                ("a".to_string(), vec![], None),
                                                (
                                                    "b".to_string(),
                                                    vec![Deriv::Ptr(TypeQualifiers::empty())],
                                                    None
                                                )
                                            ]
                                        )])
                                    )),
                                    TypeQualifiers::empty()
                                ),
                                vec![]
                            ),
                            TagItemDecl(
                                QualifiedType(
                                    UnqualifiedType::Basic(BasicType::Char),
                                    TypeQualifiers::empty()
                                ),
                                vec![(
                                    "z".to_string(),
                                    vec![Deriv::Ptr(TypeQualifiers::CONST)],
                                    None
                                )]
                            )
                        ])
                    )),
                    TypeQualifiers::empty()
                ),
                vec![("foo".to_string(), vec![])],
                None
            ))]
        );
    }
}

fn main() -> Result<(), ParseError> {
    let mut parser = Parser::from_code(r#"x;"#);
    while let Some(decl) = parser.next()? {
        println!("Declaration: {:?}", decl);
    }
    Ok(())
}
