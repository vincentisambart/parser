// TODO:
// - For testing pragmas, have a look at clang's test/Sema/pragma-align-packed.c
// - Move tests to one (or multiple) other files
mod failable;
mod lex;
mod peeking;
use bitflags::bitflags;
use crate::failable::{FailableIterator, FailablePeekable, VarRateFailableIterator};
use crate::lex::{Keyword, LexError, Position, PositionedToken, Punctuator, Token, TokenIter};
use std::collections::HashMap;

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
enum PrimitiveType {
    Void,
    Char,
    SignedChar,
    UnsignedChar,
    Short,
    UnsignedShort,
    Int,
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
enum DefinableType {
    Func(FunctionType),
    Qual(QualifiedType),
    Array(ArrayType),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QualifiableType {
    Prim(PrimitiveType),
    Ptr(Box<DefinableType>),
    Custom(String, Box<DefinableType>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QualifiedType(QualifiableType, TypeQualifiers);

#[derive(Debug, Clone, PartialEq, Eq)]
enum ContainableType {
    Qual(QualifiedType),
    Array(ArrayType),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ArraySize {
    Unspecified,
    Fixed(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ArrayType(ArraySize, Box<ContainableType>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct FunctionType(QualifiedType, FunctionParameters);

#[derive(Debug, Clone, PartialEq, Eq)]
struct FunctionParameter(Option<String>, ContainableType);

#[derive(Debug, Clone, PartialEq, Eq)]
enum FunctionParameters {
    Undefined,
    Defined {
        params: Vec<FunctionParameter>,
        variable: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum ExternalDecl {
    Decl(String, DefinableType),
    TypeDef(String, DefinableType),
}

#[derive(Debug, Clone)]
pub enum ParseError {
    LexError(LexError),
    ExpectingToken(Token), // TODO: Should have position
    UnexpectedToken(Token, Position),
    UnexpectedEOF,
}

impl From<LexError> for ParseError {
    fn from(error: LexError) -> Self {
        ParseError::LexError(error)
    }
}

struct TypeManager {
    types_stack: Vec<HashMap<String, DefinableType>>,
}

impl TypeManager {
    fn new() -> TypeManager {
        TypeManager {
            types_stack: vec![HashMap::new()],
        }
    }

    fn type_in_current_scope(&self, name: &str) -> Option<&DefinableType> {
        self.types_stack.last().unwrap().get(name)
    }

    fn add_type_to_current_scope(&mut self, name: String, ty: DefinableType) {
        if self
            .types_stack
            .last_mut()
            .unwrap()
            .insert(name, ty)
            .is_some()
        {
            panic!("You should not redefine a type already defined in the current scope");
        }
    }

    fn type_by_name(&self, name: &str) -> Option<&DefinableType> {
        for types in self.types_stack.iter().rev() {
            if let Some(ty) = types.get(name) {
                return Some(ty);
            }
        }
        None
    }

    fn is_type_name(&self, name: &str) -> bool {
        self.type_by_name(name).is_some()
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

    fn read_base_type(&mut self) -> Result<Option<QualifiableType>, ParseError> {
        let ty = if let Some(prim) = self.read_primitive_type()? {
            Some(QualifiableType::Prim(prim))
        } else {
            let type_manager = &self.type_manager;
            if let Some(PositionedToken(Token::Identifier(ident), _)) =
                self.iter.next_if(|token| match token {
                    PositionedToken(Token::Identifier(ident), _) => {
                        type_manager.is_type_name(ident.as_ref())
                    }
                    _ => false,
                })? {
                let defined_ty = type_manager.type_by_name(ident.as_ref()).unwrap().clone();
                Some(QualifiableType::Custom(ident, Box::new(defined_ty)))
            } else {
                None
            }
        };
        Ok(ty)
    }

    fn read_primitive_type(&mut self) -> Result<Option<PrimitiveType>, ParseError> {
        let ty = if self.iter.advance_if_kw(Keyword::Void)? {
            Some(PrimitiveType::Void)
        } else if self.iter.advance_if_kw(Keyword::Char)? {
            Some(PrimitiveType::Char)
        } else if self.iter.advance_if_kw(Keyword::Signed)? {
            if self.iter.advance_if_kw(Keyword::Char)? {
                Some(PrimitiveType::SignedChar)
            } else if self.iter.advance_if_kw(Keyword::Short)? {
                self.iter.advance_if_kw(Keyword::Int)?;
                Some(PrimitiveType::Short)
            } else if self.iter.advance_if_kw(Keyword::Long)? {
                if self.iter.advance_if_kw(Keyword::Long)? {
                    self.iter.advance_if_kw(Keyword::Int)?;
                    Some(PrimitiveType::LongLong)
                } else {
                    self.iter.advance_if_kw(Keyword::Int)?;
                    Some(PrimitiveType::Long)
                }
            } else {
                self.iter.advance_if_kw(Keyword::Int)?;
                Some(PrimitiveType::Int)
            }
        } else if self.iter.advance_if_kw(Keyword::Unsigned)? {
            if self.iter.advance_if_kw(Keyword::Char)? {
                Some(PrimitiveType::UnsignedChar)
            } else if self.iter.advance_if_kw(Keyword::Short)? {
                self.iter.advance_if_kw(Keyword::Int)?;
                Some(PrimitiveType::UnsignedShort)
            } else if self.iter.advance_if_kw(Keyword::Long)? {
                if self.iter.advance_if_kw(Keyword::Long)? {
                    self.iter.advance_if_kw(Keyword::Int)?;
                    Some(PrimitiveType::UnsignedLongLong)
                } else {
                    self.iter.advance_if_kw(Keyword::Int)?;
                    Some(PrimitiveType::UnsignedLong)
                }
            } else {
                self.iter.advance_if_kw(Keyword::Int)?;
                Some(PrimitiveType::UnsignedInt)
            }
        } else if self.iter.advance_if_kw(Keyword::Short)? {
            self.iter.advance_if_kw(Keyword::Int)?;
            Some(PrimitiveType::Short)
        } else if self.iter.advance_if_kw(Keyword::Int)? {
            Some(PrimitiveType::Int)
        } else if self.iter.advance_if_kw(Keyword::Long)? {
            if self.iter.advance_if_kw(Keyword::Long)? {
                self.iter.advance_if_kw(Keyword::Int)?;
                Some(PrimitiveType::LongLong)
            } else if self.iter.advance_if_kw(Keyword::Double)? {
                if self.iter.advance_if_kw(Keyword::Complex)? {
                    Some(PrimitiveType::LongDoubleComplex)
                } else {
                    Some(PrimitiveType::LongDouble)
                }
            } else {
                self.iter.advance_if_kw(Keyword::Int)?;
                Some(PrimitiveType::Long)
            }
        } else if self.iter.advance_if_kw(Keyword::Float)? {
            if self.iter.advance_if_kw(Keyword::Complex)? {
                Some(PrimitiveType::FloatComplex)
            } else {
                Some(PrimitiveType::Float)
            }
        } else if self.iter.advance_if_kw(Keyword::Double)? {
            if self.iter.advance_if_kw(Keyword::Complex)? {
                Some(PrimitiveType::DoubleComplex)
            } else {
                Some(PrimitiveType::Double)
            }
        } else if self.iter.advance_if_kw(Keyword::Bool)? {
            Some(PrimitiveType::Bool)
        } else {
            None
        };
        Ok(ty)
    }

    fn expect_token(&mut self, expected_token: Token) -> Result<(), ParseError> {
        match self.iter.next()? {
            Some(PositionedToken(token, position)) => if token == expected_token {
                Ok(())
            } else {
                eprintln!("expecting {:?}, got {:?}", expected_token, token);
                Err(ParseError::UnexpectedToken(token, position))
            },
            None => Err(ParseError::ExpectingToken(expected_token)),
        }
    }

    // Should be called just after having read an opening parenthesis.
    fn read_func_params(&mut self) -> Result<FunctionParameters, ParseError> {
        if self.iter.advance_if_punc(Punctuator::RightParenthesis)? {
            return Ok(FunctionParameters::Undefined);
        }
        // TODO: Handle parameters
        panic!("TODO: Handle parameters")
    }

    // Should be called just after having read an opening square bracket.
    fn read_array_size(&mut self) -> Result<ArraySize, ParseError> {
        if self.iter.advance_if_punc(Punctuator::RightSquareBracket)? {
            return Ok(ArraySize::Unspecified);
        }

        match self.iter.next()? {
            Some(PositionedToken(Token::IntegerLiteral(literal, repr, _), _)) => {
                let size = u32::from_str_radix(literal.as_ref(), repr.radix()).unwrap();
                self.expect_token(Token::Punctuator(Punctuator::RightSquareBracket))?;
                Ok(ArraySize::Fixed(size))
            }
            Some(PositionedToken(token, position)) => {
                Err(ParseError::UnexpectedToken(token, position))
            }
            None => Err(ParseError::UnexpectedEOF),
        }
    }
}

struct TypeDeclParenLevel {
    ptr_qualifs: Vec<TypeQualifiers>,
    func_params: Option<FunctionParameters>,
    array_sizes: Vec<ArraySize>,
}

impl TypeDeclParenLevel {
    fn new() -> TypeDeclParenLevel {
        TypeDeclParenLevel {
            ptr_qualifs: Vec::new(),
            func_params: None,
            array_sizes: Vec::new(),
        }
    }
}

impl<'a> VarRateFailableIterator for Parser<'a> {
    type Item = ExternalDecl;
    type Error = ParseError;
    type VarRateItemsIter = std::vec::IntoIter<Self::Item>;

    fn next(&mut self) -> Result<Option<Self::VarRateItemsIter>, Self::Error> {
        if self.iter.peek()?.is_none() {
            return Ok(None);
        }
        let is_typedef = self.iter.advance_if_kw(Keyword::Typedef)?;
        let base_qual_ty = {
            let mut qualifiers = TypeQualifiers::empty();
            while let Some(qualifier) = self.read_type_qualifier()? {
                qualifiers |= qualifier;
            }
            let ty = self
                .read_base_type()?
                .unwrap_or(QualifiableType::Prim(PrimitiveType::Int));
            while let Some(qualifier) = self.read_type_qualifier()? {
                qualifiers |= qualifier;
            }
            QualifiedType(ty, qualifiers)
        };

        if self.iter.advance_if_punc(Punctuator::Semicolon)? {
            return Ok(Some(Vec::new().into_iter()));
        }

        let mut decls = Vec::new();
        loop {
            let mut levels = Vec::new();
            let mut current_level = TypeDeclParenLevel::new();
            let ident = loop {
                if self.iter.advance_if_punc(Punctuator::Star)? {
                    let mut qualifiers = TypeQualifiers::empty();
                    while let Some(qualifier) = self.read_type_qualifier()? {
                        qualifiers |= qualifier;
                    }
                    current_level.ptr_qualifs.push(qualifiers);
                } else if self.iter.advance_if_punc(Punctuator::LeftParenthesis)? {
                    levels.push(current_level);
                    current_level = TypeDeclParenLevel::new();
                } else if let Some(ident) = self.iter.next_if_any_ident()? {
                    levels.push(current_level);
                    break ident;
                } else {
                    panic!("TODO: Incorrect decl");
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

            let mut def_ty = DefinableType::Qual(base_qual_ty.clone());
            for level in levels {
                for qualifiers in level.ptr_qualifs {
                    def_ty = DefinableType::Qual(QualifiedType(
                        QualifiableType::Ptr(Box::new(def_ty)),
                        qualifiers,
                    ));
                }
                if let Some(func_params) = level.func_params {
                    def_ty = match def_ty {
                        DefinableType::Qual(qual_ty) => {
                            DefinableType::Func(FunctionType(qual_ty, func_params))
                        }
                        DefinableType::Array(_) | DefinableType::Func(_) => panic!(
                            "A function can't return an array or function - TODO: proper error"
                        ),
                    };
                }
                for array_size in level.array_sizes.into_iter().rev() {
                    def_ty = match def_ty {
                        DefinableType::Qual(qual_ty) => DefinableType::Array(ArrayType(
                            array_size,
                            Box::new(ContainableType::Qual(qual_ty)),
                        )),
                        DefinableType::Array(array_ty) => DefinableType::Array(ArrayType(
                            array_size,
                            Box::new(ContainableType::Array(array_ty)),
                        )),
                        DefinableType::Func(_) => {
                            panic!("You can't have an array of funcs - TODO: proper error")
                        }
                    };
                }
            }

            if is_typedef {
                if let Some(existing) = self.type_manager.type_in_current_scope(ident.as_ref()) {
                    // TODO: Should be a comparison of the type with all custom types expanded.
                    if *existing != def_ty {
                        panic!("A typedef cannot redefine an already defined type on the same scope - TODO: proper error");
                    }
                } else {
                    self.type_manager
                        .add_type_to_current_scope(ident.clone(), def_ty.clone());
                }
                decls.push(ExternalDecl::TypeDef(ident, def_ty))
            } else {
                decls.push(ExternalDecl::Decl(ident, def_ty))
            }

            match self.iter.next()? {
                Some(PositionedToken(Token::Punctuator(Punctuator::Semicolon), _)) => break,
                Some(PositionedToken(Token::Punctuator(Punctuator::Comma), _)) => (),
                Some(PositionedToken(token, position)) => {
                    return Err(ParseError::UnexpectedToken(token, position))
                }
                None => {
                    return Err(ParseError::ExpectingToken(Token::Punctuator(
                        Punctuator::Semicolon,
                    )))
                }
            }
        }
        Ok(Some(decls.into_iter()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_external_declarations(code: &str) -> Vec<ExternalDecl> {
        let parser = Parser::from_code(code);
        match parser.collect() {
            Ok(decls) => decls,
            Err(err) => panic!(r#"Unexpected error {:?} for "{:}""#, err, code),
        }
    }

    fn parse_one_external_declaration(code: &str) -> Option<ExternalDecl> {
        let decls = parse_external_declarations(code);
        if decls.len() > 1 {
            panic!(
                r#"Unexpected {:?} after {:?} for "{:}""#,
                decls[1], decls[0], code
            );
        }
        decls.into_iter().next()
    }

    fn qual_ptr(to: DefinableType, qualifiers: TypeQualifiers) -> QualifiedType {
        QualifiedType(QualifiableType::Ptr(Box::new(to)), qualifiers)
    }

    fn def_qual_ptr(to: DefinableType, qualifiers: TypeQualifiers) -> DefinableType {
        DefinableType::Qual(qual_ptr(to, qualifiers))
    }

    fn ptr(to: DefinableType) -> QualifiedType {
        qual_ptr(to, TypeQualifiers::empty())
    }

    fn def_ptr(to: DefinableType) -> DefinableType {
        DefinableType::Qual(ptr(to))
    }

    fn func_ptr(ret_type: QualifiedType, params: FunctionParameters) -> QualifiedType {
        QualifiedType(
            QualifiableType::Ptr(Box::new(DefinableType::Func(FunctionType(
                ret_type, params,
            )))),
            TypeQualifiers::empty(),
        )
    }

    fn def_func_ptr(ret_type: QualifiedType, params: FunctionParameters) -> DefinableType {
        DefinableType::Qual(func_ptr(ret_type, params))
    }

    #[test]
    fn test_simple_declaration() {
        assert_eq!(parse_one_external_declaration(r#""#), None);
        assert_eq!(
            parse_one_external_declaration(r#"abcd;"#),
            Some(ExternalDecl::Decl(
                "abcd".to_string(),
                DefinableType::Qual(QualifiedType(
                    QualifiableType::Prim(PrimitiveType::Int),
                    TypeQualifiers::empty()
                ))
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"(abcd);"#),
            Some(ExternalDecl::Decl(
                "abcd".to_string(),
                DefinableType::Qual(QualifiedType(
                    QualifiableType::Prim(PrimitiveType::Int),
                    TypeQualifiers::empty()
                )),
            )),
        );
        assert_eq!(
            parse_one_external_declaration(r#"int abcd;"#),
            Some(ExternalDecl::Decl(
                "abcd".to_string(),
                DefinableType::Qual(QualifiedType(
                    QualifiableType::Prim(PrimitiveType::Int),
                    TypeQualifiers::empty()
                )),
            ))
        );
        assert_eq!(parse_external_declarations(r#"signed long;"#), vec![]);
        assert_eq!(
            parse_one_external_declaration(r#"unsigned char abcd;"#),
            Some(ExternalDecl::Decl(
                "abcd".to_string(),
                DefinableType::Qual(QualifiedType(
                    QualifiableType::Prim(PrimitiveType::UnsignedChar),
                    TypeQualifiers::empty()
                ))
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"*abcd;"#),
            Some(ExternalDecl::Decl(
                "abcd".to_string(),
                def_ptr(DefinableType::Qual(QualifiedType(
                    QualifiableType::Prim(PrimitiveType::Int),
                    TypeQualifiers::empty()
                )))
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"signed long long int * abcd;"#),
            Some(ExternalDecl::Decl(
                "abcd".to_string(),
                def_ptr(DefinableType::Qual(QualifiedType(
                    QualifiableType::Prim(PrimitiveType::LongLong),
                    TypeQualifiers::empty()
                )))
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"const short * abcd;"#),
            Some(ExternalDecl::Decl(
                "abcd".to_string(),
                def_ptr(DefinableType::Qual(QualifiedType(
                    QualifiableType::Prim(PrimitiveType::Short),
                    TypeQualifiers::CONST
                )))
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"short const * abcd;"#),
            Some(ExternalDecl::Decl(
                "abcd".to_string(),
                def_ptr(DefinableType::Qual(QualifiedType(
                    QualifiableType::Prim(PrimitiveType::Short),
                    TypeQualifiers::CONST
                )))
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"float * const abcd;"#),
            Some(ExternalDecl::Decl(
                "abcd".to_string(),
                def_qual_ptr(
                    DefinableType::Qual(QualifiedType(
                        QualifiableType::Prim(PrimitiveType::Float),
                        TypeQualifiers::empty()
                    )),
                    TypeQualifiers::CONST
                )
            ))
        );
    }

    #[test]
    fn test_parse_function_declaration() {
        assert_eq!(
            parse_one_external_declaration(r#"foo();"#),
            Some(ExternalDecl::Decl(
                "foo".to_string(),
                DefinableType::Func(FunctionType(
                    QualifiedType(
                        QualifiableType::Prim(PrimitiveType::Int),
                        TypeQualifiers::empty()
                    ),
                    FunctionParameters::Undefined
                ))
            ))
        );
    }

    #[test]
    fn test_parse_function_pointer_declaration() {
        assert_eq!(
            parse_one_external_declaration(r#"int (*foo)();"#),
            Some(ExternalDecl::Decl(
                "foo".to_string(),
                def_func_ptr(
                    QualifiedType(
                        QualifiableType::Prim(PrimitiveType::Int),
                        TypeQualifiers::empty()
                    ),
                    FunctionParameters::Undefined
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"int (*(foo))();"#),
            Some(ExternalDecl::Decl(
                "foo".to_string(),
                def_func_ptr(
                    QualifiedType(
                        QualifiableType::Prim(PrimitiveType::Int),
                        TypeQualifiers::empty()
                    ),
                    FunctionParameters::Undefined
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"int (*(*bar)())();"#),
            Some(ExternalDecl::Decl(
                "bar".to_string(),
                def_func_ptr(
                    func_ptr(
                        QualifiedType(
                            QualifiableType::Prim(PrimitiveType::Int),
                            TypeQualifiers::empty()
                        ),
                        FunctionParameters::Undefined
                    ),
                    FunctionParameters::Undefined
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"int (*foo())();"#),
            Some(ExternalDecl::Decl(
                "foo".to_string(),
                DefinableType::Func(FunctionType(
                    func_ptr(
                        QualifiedType(
                            QualifiableType::Prim(PrimitiveType::Int),
                            TypeQualifiers::empty()
                        ),
                        FunctionParameters::Undefined
                    ),
                    FunctionParameters::Undefined
                ))
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"char const * (**hogehoge)();"#),
            Some(ExternalDecl::Decl(
                "hogehoge".to_string(),
                def_ptr(def_func_ptr(
                    ptr(DefinableType::Qual(QualifiedType(
                        QualifiableType::Prim(PrimitiveType::Char),
                        TypeQualifiers::CONST
                    ))),
                    FunctionParameters::Undefined
                ))
            ))
        );
    }

    #[test]
    fn test_parse_array_declaration() {
        assert_eq!(
            parse_one_external_declaration(r#"int foo[10];"#),
            Some(ExternalDecl::Decl(
                "foo".to_string(),
                DefinableType::Array(ArrayType(
                    ArraySize::Fixed(10),
                    Box::new(ContainableType::Qual(QualifiedType(
                        QualifiableType::Prim(PrimitiveType::Int),
                        TypeQualifiers::empty()
                    )))
                ))
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"char foo[1][3];"#),
            Some(ExternalDecl::Decl(
                "foo".to_string(),
                DefinableType::Array(ArrayType(
                    ArraySize::Fixed(1),
                    Box::new(ContainableType::Array(ArrayType(
                        ArraySize::Fixed(3),
                        Box::new(ContainableType::Qual(QualifiedType(
                            QualifiableType::Prim(PrimitiveType::Char),
                            TypeQualifiers::empty()
                        )))
                    )))
                ))
            ))
        );
        assert_eq!(
            // TODO: Unspecified size arrays should only be used in function parameters
            parse_one_external_declaration(r#"short foo[];"#),
            Some(ExternalDecl::Decl(
                "foo".to_string(),
                DefinableType::Array(ArrayType(
                    ArraySize::Unspecified,
                    Box::new(ContainableType::Qual(QualifiedType(
                        QualifiableType::Prim(PrimitiveType::Short),
                        TypeQualifiers::empty()
                    )))
                ))
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"char *(*abcdef[3])();"#),
            Some(ExternalDecl::Decl(
                "abcdef".to_string(),
                DefinableType::Array(ArrayType(
                    ArraySize::Fixed(3),
                    Box::new(ContainableType::Qual(func_ptr(
                        ptr(DefinableType::Qual(QualifiedType(
                            QualifiableType::Prim(PrimitiveType::Char),
                            TypeQualifiers::empty()
                        ))),
                        FunctionParameters::Undefined
                    )))
                ))
            ))
        );
    }

    #[test]
    fn test_simple_type_definition() {
        assert_eq!(
            parse_one_external_declaration(r#"typedef signed *truc();"#),
            Some(ExternalDecl::TypeDef(
                "truc".to_string(),
                DefinableType::Func(FunctionType(
                    ptr(DefinableType::Qual(QualifiedType(
                        QualifiableType::Prim(PrimitiveType::Int),
                        TypeQualifiers::empty()
                    ))),
                    FunctionParameters::Undefined
                ))
            ))
        );
        assert_eq!(
            parse_external_declarations(r#"typedef int *ptr; const ptr foo;"#),
            vec![
                ExternalDecl::TypeDef(
                    "ptr".to_string(),
                    def_ptr(DefinableType::Qual(QualifiedType(
                        QualifiableType::Prim(PrimitiveType::Int),
                        TypeQualifiers::empty()
                    ))),
                ),
                ExternalDecl::Decl(
                    "foo".to_string(),
                    DefinableType::Qual(QualifiedType(
                        QualifiableType::Custom(
                            "ptr".to_string(),
                            Box::new(def_ptr(DefinableType::Qual(QualifiedType(
                                QualifiableType::Prim(PrimitiveType::Int),
                                TypeQualifiers::empty()
                            ))))
                        ),
                        TypeQualifiers::CONST
                    ))
                )
            ],
        );
        assert_eq!(
            parse_external_declarations(r#"typedef int i, *ptr; ptr foo, *bar;"#),
            vec![
                ExternalDecl::TypeDef(
                    "i".to_string(),
                    DefinableType::Qual(QualifiedType(
                        QualifiableType::Prim(PrimitiveType::Int),
                        TypeQualifiers::empty()
                    )),
                ),
                ExternalDecl::TypeDef(
                    "ptr".to_string(),
                    def_ptr(DefinableType::Qual(QualifiedType(
                        QualifiableType::Prim(PrimitiveType::Int),
                        TypeQualifiers::empty()
                    ))),
                ),
                ExternalDecl::Decl(
                    "foo".to_string(),
                    DefinableType::Qual(QualifiedType(
                        QualifiableType::Custom(
                            "ptr".to_string(),
                            Box::new(def_ptr(DefinableType::Qual(QualifiedType(
                                QualifiableType::Prim(PrimitiveType::Int),
                                TypeQualifiers::empty()
                            ))))
                        ),
                        TypeQualifiers::empty()
                    ))
                ),
                ExternalDecl::Decl(
                    "bar".to_string(),
                    def_ptr(DefinableType::Qual(QualifiedType(
                        QualifiableType::Custom(
                            "ptr".to_string(),
                            Box::new(def_ptr(DefinableType::Qual(QualifiedType(
                                QualifiableType::Prim(PrimitiveType::Int),
                                TypeQualifiers::empty()
                            ))))
                        ),
                        TypeQualifiers::empty()
                    )))
                )
            ]
        );
        // assert_eq!(
        //     parse_external_declarations(r#"typedef int *foo, bar(foo x);"#),
        //     vec![
        //         ExternalDecl::TypeDef(
        //             "foo".to_string(),
        //             def_ptr(DefinableType::Qual(QualifiedType(
        //                 QualifiableType::Prim(PrimitiveType::Int),
        //                 TypeQualifiers::empty()
        //             )))
        //         ),
        //         ExternalDecl::TypeDef(
        //             "bar".to_string(),
        //             DefinableType::Func(FunctionType(
        //                 QualifiedType(
        //                     QualifiableType::Prim(PrimitiveType::Int),
        //                     TypeQualifiers::empty()
        //                 ),
        //                 FunctionParameters::Defined {
        //                     params: vec![FunctionParameter(
        //                         Some("x".to_string()),
        //                         ContainableType::Qual(QualifiedType(
        //                             QualifiableType::Custom(
        //                                 "foo".to_string(),
        //                                 Box::new(def_ptr(DefinableType::Qual(QualifiedType(
        //                                     QualifiableType::Prim(PrimitiveType::Int),
        //                                     TypeQualifiers::empty()
        //                                 ))))
        //                             ),
        //                             TypeQualifiers::empty()
        //                         )),
        //                     )],
        //                     variable: true,
        //                 }
        //             ))
        //         )
        //     ]
        // );
    }
}

fn main() -> Result<(), ParseError> {
    let mut parser = Parser::from_code(r#"x;"#);
    let decls = parser.next()?;
    println!("Declaration: {:?}", decls);
    Ok(())
}
