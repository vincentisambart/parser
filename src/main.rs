// TODO:
// - For testing pragmas, have a look at clang's test/Sema/pragma-align-packed.c
// - Move tests to one (or multiple) other files
// - Add position to declarations
// - Make most panics/expect normal errors
// - Linkage (extern/static)
// - Base type, storage class (including typedef), function specifier, can be specified in any order
// - struct/enum/union
// - Function definition
// - Variable initialization

mod error;
mod failable;
mod lex;
mod peeking;

use crate::error::{ParseError, ParseErrorKind};
use crate::failable::{FailableIterator, FailablePeekable};
use crate::lex::{Keyword, PositionedToken, Punctuator, Token, TokenIter};

use bitflags::bitflags;
use std::collections::HashSet;

trait PeekingToken {
    type Error;

    fn advance_if_kw(&mut self, kw: Keyword) -> Result<bool, Self::Error>;
    fn advance_if_punc(&mut self, punc: Punctuator) -> Result<bool, Self::Error>;
    fn next_if_any_ident(&mut self) -> Result<Option<String>, Self::Error>;
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

    fn next_if_any_ident(&mut self) -> Result<Option<String>, I::Error> {
        let token = self.next_if(|token| match token {
            PositionedToken(Token::Identifier(_), _) => true,
            _ => false,
        })?;
        match token {
            Some(PositionedToken(Token::Identifier(ident), _)) => Ok(Some(ident)),
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

    fn read_type_specifier(&mut self) -> Result<Option<UnqualifiedType>, ParseError> {
        let ty = if let Some(prim) = self.read_basic_type()? {
            Some(UnqualifiedType::Basic(prim))
        } else if self.iter.advance_if_kw(Keyword::Enum)? {
            let enum_name = self.iter.next_if_any_ident()?;
            if self.iter.advance_if_punc(Punctuator::LeftCurlyBracket)? {
                let mut enum_values = Vec::new();
                loop {
                    if self.iter.advance_if_punc(Punctuator::RightCurlyBracket)? {
                        break;
                    } else if let Some(value_name) = self.iter.next_if_any_ident()? {
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
                Some(UnqualifiedType::Tag(
                    enum_name,
                    Tag::Enum(Some(enum_values)),
                ))
            } else {
                Some(UnqualifiedType::Tag(enum_name, Tag::Enum(None)))
            }
        } else {
            let type_manager = &self.type_manager;
            if let Some(PositionedToken(Token::Identifier(ident), _)) =
                self.iter.next_if(|token| match token {
                    PositionedToken(Token::Identifier(ident), _) => {
                        type_manager.is_type_name(ident.as_ref())
                    }
                    _ => false,
                })? {
                Some(UnqualifiedType::Custom(ident))
            } else {
                None
            }
        };
        Ok(ty)
    }

    fn read_basic_type(&mut self) -> Result<Option<BasicType>, ParseError> {
        let ty = if self.iter.advance_if_kw(Keyword::Void)? {
            Some(BasicType::Void)
        } else if self.iter.advance_if_kw(Keyword::Char)? {
            Some(BasicType::Char)
        } else if self.iter.advance_if_kw(Keyword::Signed)? {
            if self.iter.advance_if_kw(Keyword::Char)? {
                Some(BasicType::SignedChar)
            } else if self.iter.advance_if_kw(Keyword::Short)? {
                self.iter.advance_if_kw(Keyword::Int)?;
                Some(BasicType::Short)
            } else if self.iter.advance_if_kw(Keyword::Long)? {
                if self.iter.advance_if_kw(Keyword::Long)? {
                    self.iter.advance_if_kw(Keyword::Int)?;
                    Some(BasicType::LongLong)
                } else {
                    self.iter.advance_if_kw(Keyword::Int)?;
                    Some(BasicType::Long)
                }
            } else {
                self.iter.advance_if_kw(Keyword::Int)?;
                Some(BasicType::SignedInt)
            }
        } else if self.iter.advance_if_kw(Keyword::Unsigned)? {
            if self.iter.advance_if_kw(Keyword::Char)? {
                Some(BasicType::UnsignedChar)
            } else if self.iter.advance_if_kw(Keyword::Short)? {
                self.iter.advance_if_kw(Keyword::Int)?;
                Some(BasicType::UnsignedShort)
            } else if self.iter.advance_if_kw(Keyword::Long)? {
                if self.iter.advance_if_kw(Keyword::Long)? {
                    self.iter.advance_if_kw(Keyword::Int)?;
                    Some(BasicType::UnsignedLongLong)
                } else {
                    self.iter.advance_if_kw(Keyword::Int)?;
                    Some(BasicType::UnsignedLong)
                }
            } else {
                self.iter.advance_if_kw(Keyword::Int)?;
                Some(BasicType::UnsignedInt)
            }
        } else if self.iter.advance_if_kw(Keyword::Short)? {
            self.iter.advance_if_kw(Keyword::Int)?;
            Some(BasicType::Short)
        } else if self.iter.advance_if_kw(Keyword::Int)? {
            Some(BasicType::Int)
        } else if self.iter.advance_if_kw(Keyword::Long)? {
            if self.iter.advance_if_kw(Keyword::Long)? {
                self.iter.advance_if_kw(Keyword::Int)?;
                Some(BasicType::LongLong)
            } else if self.iter.advance_if_kw(Keyword::Double)? {
                if self.iter.advance_if_kw(Keyword::Complex)? {
                    Some(BasicType::LongDoubleComplex)
                } else {
                    Some(BasicType::LongDouble)
                }
            } else {
                self.iter.advance_if_kw(Keyword::Int)?;
                Some(BasicType::Long)
            }
        } else if self.iter.advance_if_kw(Keyword::Float)? {
            if self.iter.advance_if_kw(Keyword::Complex)? {
                Some(BasicType::FloatComplex)
            } else {
                Some(BasicType::Float)
            }
        } else if self.iter.advance_if_kw(Keyword::Double)? {
            if self.iter.advance_if_kw(Keyword::Complex)? {
                Some(BasicType::DoubleComplex)
            } else {
                Some(BasicType::Double)
            }
        } else if self.iter.advance_if_kw(Keyword::Bool)? {
            Some(BasicType::Bool)
        } else {
            None
        };
        Ok(ty)
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
            let decl_spec = self.read_declaration_specifier()?;
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
                break self.iter.next_if_any_ident()?;
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

    fn read_declaration_specifier(&mut self) -> Result<Option<QualifiedType>, ParseError> {
        let mut qualifiers = TypeQualifiers::empty();
        while let Some(qualifier) = self.read_type_qualifier()? {
            qualifiers |= qualifier;
        }
        let base_ty = self.read_type_specifier()?;
        while let Some(qualifier) = self.read_type_qualifier()? {
            qualifiers |= qualifier;
        }
        let base_ty = match base_ty {
            Some(ty) => Some(QualifiedType(ty, qualifiers)),
            None => {
                if qualifiers.is_empty() {
                    None
                } else {
                    Some(QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        qualifiers,
                    ))
                }
            }
        };
        Ok(base_ty)
    }
}

impl<'a> FailableIterator for Parser<'a> {
    type Item = ExtDecl;
    type Error = ParseError;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        if self.iter.peek()?.is_none() {
            return Ok(None);
        }
        let is_typedef = self.iter.advance_if_kw(Keyword::Typedef)?;
        let decl_spec = self.read_declaration_specifier()?.unwrap_or_else(|| {
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
        // assert_eq!(
        //     // typedef, basic types keywords can be in any order 😢
        //     parse_external_declarations(r#"long typedef long unsigned foo;"#),
        //     vec![ExtDecl::TypeDef(Decl(
        //         QualifiedType(
        //             UnqualifiedType::Basic(BasicType::UnsignedLongLong),
        //             TypeQualifiers::empty()
        //         ),
        //         vec![("foo".to_string(), vec![Deriv::Ptr(TypeQualifiers::empty())])]
        //     ))]
        // );
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
