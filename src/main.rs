// Notes
// - For testing pragmas, have a look at clang's test/Sema/pragma-align-packed.c
mod lex;
mod scan;
use bitflags::bitflags;
use crate::lex::{Keyword, LexError, Position, PositionedToken, Punctuator, Token, TokenIter};
use crate::scan::Peeking;
use std::collections::HashMap;
use std::iter::Peekable;
use std::rc::Rc;

trait PeekingToken {
    fn advance_if_kw(&mut self, kw: Keyword) -> Result<bool, LexError>;
    fn advance_if_punc(&mut self, punc: Punctuator) -> Result<bool, LexError>;
    fn next_if_any_ident(&mut self) -> Result<Option<String>, LexError>;
}

impl<I> PeekingToken for Peekable<I>
where
    I: Iterator<Item = Result<PositionedToken, LexError>>,
{
    fn advance_if_kw(&mut self, kw: Keyword) -> Result<bool, LexError> {
        let token = self.next_if_lifting_err(|token| match token.token() {
            Token::Keyword(x) if x == kw => true,
            _ => false,
        })?;
        Ok(token.is_some())
    }

    fn advance_if_punc(&mut self, punc: Punctuator) -> Result<bool, LexError> {
        let token = self.next_if_lifting_err(|token| match token.token() {
            Token::Punctuator(x) if x == punc => true,
            _ => false,
        })?;
        Ok(token.is_some())
    }

    fn next_if_any_ident(&mut self) -> Result<Option<String>, LexError> {
        let token = self.next_if_lifting_err(|token| match token.token() {
            Token::Identifier(_) => true,
            _ => false,
        })?;
        match token {
            Some(token) => match token.token() {
                Token::Identifier(ident) => Ok(Some(ident)),
                _ => unreachable!(),
            },
            None => Ok(None),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
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
        const RESTRICT = 1 << 2;
        const ATOMIC   = 1 << 3;
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Type {
    Primitive(PrimitiveType),
    Pointer(Box<QualType>),
    FuncPointer(Box<FuncType>),
    UserDefined(String, Rc<QualType>),
}

#[derive(Debug, Clone, PartialEq)]
struct QualType {
    ty: Type,
    qualifiers: TypeQualifiers,
}

impl QualType {
    fn new(ty: Type, qualifiers: TypeQualifiers) -> QualType {
        QualType { ty, qualifiers }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum FuncArgs {
    Undefined,
    Defined {
        args: Vec<(String, Type)>,
        var_args: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
struct FuncType {
    ret_type: QualType,
    args: FuncArgs,
}

impl FuncType {
    fn new(ret_type: QualType, args: FuncArgs) -> FuncType {
        FuncType { ret_type, args }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ExternalDecl {
    VarDecl(String, QualType),
    FuncDef(String, FuncType),
    Nothing,
}

#[derive(Debug, Clone)]
pub enum ParseError {
    LexError(LexError),
    ExpectingToken(Token), // TODO: Should have position
    UnexpectedToken(Token, Position),
}

impl From<LexError> for ParseError {
    fn from(error: LexError) -> Self {
        ParseError::LexError(error)
    }
}

struct Parser<'a> {
    iter: Peekable<TokenIter<'a>>,
    types_stack: Vec<HashMap<String, Rc<Type>>>,
}

impl<'a> Parser<'a> {
    fn from(code: &'a str) -> Parser<'a> {
        let iter = TokenIter::from(code).peekable();
        Parser {
            iter,
            types_stack: Vec::new(),
        }
    }

    fn type_by_name(&self, name: &str) -> Option<Rc<Type>> {
        for types in self.types_stack.iter().rev() {
            if let Some(ty) = types.get(name) {
                return Some(ty.clone());
            }
        }
        None
    }

    fn is_type_name(&self, name: &str) -> bool {
        self.type_by_name(name).is_some()
    }

    fn read_type_qualifier(&mut self) -> Result<Option<TypeQualifiers>, ParseError> {
        if self.iter.advance_if_kw(Keyword::Const)? {
            Ok(Some(TypeQualifiers::CONST))
        } else if self.iter.advance_if_kw(Keyword::Volatile)? {
            Ok(Some(TypeQualifiers::VOLATILE))
        } else if self.iter.advance_if_kw(Keyword::Restrict)? {
            Ok(Some(TypeQualifiers::RESTRICT))
        } else if self.iter.advance_if_kw(Keyword::Atomic)? {
            Ok(Some(TypeQualifiers::ATOMIC))
        } else {
            Ok(None)
        }
    }

    fn read_base_type(&mut self) -> Result<Option<Type>, ParseError> {
        if let Some(prim) = self.read_primitive_type()? {
            return Ok(Some(Type::Primitive(prim)));
        }
        let token = match self.iter.peek() {
            Some(Ok(token)) => token,
            Some(Err(_)) => {
                let err = self.iter.next().unwrap().err().unwrap().into();
                return Err(err);
            }
            None => return Ok(None),
        };
        match token.token() {
            Token::Identifier(identifier) => if self.is_type_name(identifier.as_ref()) {
                panic!("TODO")
            } else {
                Ok(None)
            },
            Token::Keyword(Keyword::Int) => {
                self.iter.next();
                Ok(Some(Type::Primitive(PrimitiveType::Int)))
            }
            _ => Ok(None),
        }
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
        match self.iter.next() {
            Some(Ok(token)) => if token.token() == expected_token {
                Ok(())
            } else {
                Err(ParseError::UnexpectedToken(token.token(), token.position()))
            },
            Some(Err(err)) => Err(err.into()),
            None => Err(ParseError::ExpectingToken(expected_token)),
        }
    }

    // Should be called just after having read an opening parenthesis.
    fn read_func_args(&mut self) -> Result<FuncArgs, ParseError> {
        // TODO: Handle parameters
        self.expect_token(Token::Punctuator(Punctuator::RightParenthesis))?;
        Ok(FuncArgs::Undefined)
    }

    fn read_next_external_declaration(&mut self) -> Result<Option<ExternalDecl>, ParseError> {
        if self.iter.peek().is_none() {
            return Ok(None);
        }
        let base_qual_ty = {
            let mut qualifiers = TypeQualifiers::empty();
            while let Some(qualifier) = self.read_type_qualifier()? {
                qualifiers |= qualifier;
            }
            let ty = self
                .read_base_type()?
                .unwrap_or(Type::Primitive(PrimitiveType::Int));
            while let Some(qualifier) = self.read_type_qualifier()? {
                qualifiers |= qualifier;
            }
            QualType::new(ty, qualifiers)
        };
        let mut ptr_qualifs = Vec::new();
        let mut ptr_qualifs_stack = Vec::new();
        let ident = loop {
            if self.iter.advance_if_punc(Punctuator::Star)? {
                let mut qualifiers = TypeQualifiers::empty();
                while let Some(qualifier) = self.read_type_qualifier()? {
                    qualifiers |= qualifier;
                }
                ptr_qualifs.push(qualifiers);
            } else if self.iter.advance_if_punc(Punctuator::LeftParenthesis)? {
                ptr_qualifs_stack.push(ptr_qualifs);
                ptr_qualifs = Vec::new();
            } else if let Some(ident) = self.iter.next_if_any_ident()? {
                ptr_qualifs_stack.push(ptr_qualifs);
                break ident;
            } else if self.iter.advance_if_punc(Punctuator::Semicolon)? {
                if !ptr_qualifs_stack.is_empty() {
                    panic!("TODO");
                }
                return Ok(Some(ExternalDecl::Nothing));
            } else {
                self.expect_token(Token::Punctuator(Punctuator::Semicolon))?;
            }
        };
        let mut ptr_qualifs_reversed_stack = Vec::new();
        loop {
            let ptr_qualifs = if let Some(ptr_qualifs) = ptr_qualifs_stack.pop() {
                ptr_qualifs
            } else {
                panic!("TODO: Incorrect decl");
            };
            if self.iter.advance_if_punc(Punctuator::LeftParenthesis)? {
                let func_args = self.read_func_args()?;
                ptr_qualifs_reversed_stack.push((ptr_qualifs, Some(func_args)));
            } else {
                ptr_qualifs_reversed_stack.push((ptr_qualifs, None));
            }
            if ptr_qualifs_stack.is_empty() {
                break;
            } else {
                self.expect_token(Token::Punctuator(Punctuator::RightParenthesis))?;
            }
        }
        assert!(ptr_qualifs_stack.is_empty());
        drop(ptr_qualifs_stack);

        self.expect_token(Token::Punctuator(Punctuator::Semicolon))?;

        enum FuncOrQualType {
            Func(FuncType),
            QualType(QualType),
        }

        let mut func_or_qual_ty = FuncOrQualType::QualType(base_qual_ty);
        while let Some((ptr_qualifs, func_args)) = ptr_qualifs_reversed_stack.pop() {
            for qualifiers in ptr_qualifs {
                func_or_qual_ty = match func_or_qual_ty {
                    FuncOrQualType::QualType(qual_ty) => FuncOrQualType::QualType(QualType::new(
                        Type::Pointer(Box::new(qual_ty)),
                        qualifiers,
                    )),
                    FuncOrQualType::Func(func) => FuncOrQualType::QualType(QualType::new(
                        Type::FuncPointer(Box::new(func)),
                        qualifiers,
                    )),
                };
            }
            if let Some(func_args) = func_args {
                func_or_qual_ty = match func_or_qual_ty {
                    FuncOrQualType::QualType(qual_ty) => {
                        FuncOrQualType::Func(FuncType::new(qual_ty, func_args))
                    }
                    FuncOrQualType::Func(_) => panic!("bad"),
                };
            }
        }

        match func_or_qual_ty {
            FuncOrQualType::QualType(qual_ty) => Ok(Some(ExternalDecl::VarDecl(ident, qual_ty))),
            FuncOrQualType::Func(func) => Ok(Some(ExternalDecl::FuncDef(ident, func))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_one_external_declaration(code: &str) -> Option<ExternalDecl> {
        let mut parser = Parser::from(code);
        let decl = match parser.read_next_external_declaration() {
            Ok(Some(decl)) => decl,
            Ok(None) => return None,
            Err(err) => panic!(r#"Unexpected error {:?} for "{:}""#, err, code),
        };
        match parser.read_next_external_declaration() {
            Ok(None) => Some(decl),
            Ok(Some(another)) => panic!(
                r#"Unexpected {:?} after {:?} for "{:}""#,
                another, decl, code
            ),
            Err(err) => panic!(
                r#"Unexpected error {:?} after {:?} for "{:}""#,
                err, decl, code
            ),
        }
    }

    #[test]
    fn test_simple_declaration() {
        assert_eq!(parse_one_external_declaration(r#""#), None);
        assert_eq!(
            parse_one_external_declaration(r#"abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                QualType::new(Type::Primitive(PrimitiveType::Int), TypeQualifiers::empty())
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"(abcd);"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                QualType::new(Type::Primitive(PrimitiveType::Int), TypeQualifiers::empty()),
            )),
        );
        assert_eq!(
            parse_one_external_declaration(r#"int abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                QualType::new(Type::Primitive(PrimitiveType::Int), TypeQualifiers::empty()),
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"signed long;"#),
            Some(ExternalDecl::Nothing)
        );
        assert_eq!(
            parse_one_external_declaration(r#"unsigned char abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                QualType::new(
                    Type::Primitive(PrimitiveType::UnsignedChar),
                    TypeQualifiers::empty()
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"*abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                QualType::new(
                    Type::Pointer(Box::new(QualType::new(
                        Type::Primitive(PrimitiveType::Int),
                        TypeQualifiers::empty()
                    ))),
                    TypeQualifiers::empty()
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"signed long long int * abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                QualType::new(
                    Type::Pointer(Box::new(QualType::new(
                        Type::Primitive(PrimitiveType::LongLong),
                        TypeQualifiers::empty()
                    ))),
                    TypeQualifiers::empty()
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"const short * abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                QualType::new(
                    Type::Pointer(Box::new(QualType::new(
                        Type::Primitive(PrimitiveType::Short),
                        TypeQualifiers::CONST
                    ))),
                    TypeQualifiers::empty()
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"short const * abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                QualType::new(
                    Type::Pointer(Box::new(QualType::new(
                        Type::Primitive(PrimitiveType::Short),
                        TypeQualifiers::CONST
                    ))),
                    TypeQualifiers::empty()
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"float * const abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                QualType::new(
                    Type::Pointer(Box::new(QualType::new(
                        Type::Primitive(PrimitiveType::Float),
                        TypeQualifiers::empty()
                    ))),
                    TypeQualifiers::CONST
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"foo();"#),
            Some(ExternalDecl::FuncDef(
                "foo".to_string(),
                FuncType::new(
                    QualType::new(Type::Primitive(PrimitiveType::Int), TypeQualifiers::empty()),
                    FuncArgs::Undefined
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"int (*foo)();"#),
            Some(ExternalDecl::VarDecl(
                "foo".to_string(),
                QualType::new(
                    Type::FuncPointer(Box::new(FuncType::new(
                        QualType::new(Type::Primitive(PrimitiveType::Int), TypeQualifiers::empty()),
                        FuncArgs::Undefined
                    ))),
                    TypeQualifiers::empty()
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"int (*(foo))();"#),
            Some(ExternalDecl::VarDecl(
                "foo".to_string(),
                QualType::new(
                    Type::FuncPointer(Box::new(FuncType::new(
                        QualType::new(Type::Primitive(PrimitiveType::Int), TypeQualifiers::empty()),
                        FuncArgs::Undefined
                    ))),
                    TypeQualifiers::empty()
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"int (*(*bar)())();"#),
            Some(ExternalDecl::VarDecl(
                "bar".to_string(),
                QualType::new(
                    Type::FuncPointer(Box::new(FuncType::new(
                        QualType::new(
                            Type::FuncPointer(Box::new(FuncType::new(
                                QualType::new(
                                    Type::Primitive(PrimitiveType::Int),
                                    TypeQualifiers::empty()
                                ),
                                FuncArgs::Undefined
                            ))),
                            TypeQualifiers::empty()
                        ),
                        FuncArgs::Undefined
                    ))),
                    TypeQualifiers::empty()
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"int (*foo())();"#),
            Some(ExternalDecl::FuncDef(
                "foo".to_string(),
                FuncType::new(
                    QualType::new(
                        Type::FuncPointer(Box::new(FuncType::new(
                            QualType::new(
                                Type::Primitive(PrimitiveType::Int),
                                TypeQualifiers::empty()
                            ),
                            FuncArgs::Undefined
                        ))),
                        TypeQualifiers::empty()
                    ),
                    FuncArgs::Undefined
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"char const * (**hogehoge)();"#),
            Some(ExternalDecl::VarDecl(
                "hogehoge".to_string(),
                QualType::new(
                    Type::Pointer(Box::new(QualType::new(
                        Type::FuncPointer(Box::new(FuncType::new(
                            QualType::new(
                                Type::Pointer(Box::new(QualType::new(
                                    Type::Primitive(PrimitiveType::Char),
                                    TypeQualifiers::CONST
                                ))),
                                TypeQualifiers::empty()
                            ),
                            FuncArgs::Undefined
                        ))),
                        TypeQualifiers::empty()
                    ))),
                    TypeQualifiers::empty()
                )
            ))
        );
    }
}

fn main() -> Result<(), LexError> {
    let mut parser = Parser::from(r#"x;"#);
    let decl = parser.read_next_external_declaration();
    println!("Declaration: {:?}", decl);
    Ok(())
}
