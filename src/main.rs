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
    fn advance_if_kw(&mut self, kw: Keyword) -> bool;
    fn advance_if_punc(&mut self, punc: Punctuator) -> bool;
    fn next_if_any_ident(&mut self) -> Option<String>;
}

impl<I> PeekingToken for Peekable<I>
where
    I: Iterator<Item = PositionedToken>,
{
    fn advance_if_kw(&mut self, kw: Keyword) -> bool {
        self.advance_if(|token| match token {
            PositionedToken(Token::Keyword(x), _) if *x == kw => true,
            _ => false,
        })
    }

    fn advance_if_punc(&mut self, punc: Punctuator) -> bool {
        self.advance_if(|token| match token {
            PositionedToken(Token::Punctuator(x), _) if *x == punc => true,
            _ => false,
        })
    }

    fn next_if_any_ident(&mut self) -> Option<String> {
        let token = self.next_if(|token| match token {
            PositionedToken(Token::Identifier(_), _) => true,
            _ => false,
        });
        match token {
            Some(PositionedToken(Token::Identifier(ident), _)) => Some(ident),
            Some(_) => unreachable!(),
            None => None,
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
struct ArrayType();

#[derive(Debug, Clone, PartialEq, Eq)]
enum DefinableType {
    Func(FunctionType),
    Qual(QualifiedType),
    Array(ArrayType),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ContainableType {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FunctionType(QualifiedType, FunctionParameters);

#[derive(Debug, Clone, PartialEq, Eq)]
enum QualifiableType {
    Prim(PrimitiveType),
    Ptr(Box<DefinableType>),
    Custom(String, Rc<DefinableType>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QualifiedType(QualifiableType, TypeQualifiers);

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

struct Parser<I: Iterator<Item = PositionedToken>> {
    iter: Peekable<I>,
    types_stack: Vec<HashMap<String, Rc<DefinableType>>>,
}

impl Parser<std::vec::IntoIter<PositionedToken>> {
    fn from_code(code: &str) -> Result<Self, ParseError> {
        let iter = TokenIter::from(code);
        let mut tokens = Vec::new();
        for result in iter {
            match result {
                Ok(token) => tokens.push(token),
                Err(err) => return Err(err.into()),
            }
        }
        Ok(Self::from(tokens.into_iter()))
    }
}

impl<I> Parser<I>
where
    I: Iterator<Item = PositionedToken>,
{
    fn from(iter: I) -> Parser<I> {
        Parser {
            iter: iter.peekable(),
            types_stack: Vec::new(),
        }
    }

    fn type_by_name(&self, name: &str) -> Option<Rc<DefinableType>> {
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

    fn read_type_qualifier(&mut self) -> Option<TypeQualifiers> {
        if self.iter.advance_if_kw(Keyword::Const) {
            Some(TypeQualifiers::CONST)
        } else if self.iter.advance_if_kw(Keyword::Volatile) {
            Some(TypeQualifiers::VOLATILE)
        } else if self.iter.advance_if_kw(Keyword::Restrict) {
            Some(TypeQualifiers::RESTRICT)
        } else if self.iter.advance_if_kw(Keyword::Atomic) {
            Some(TypeQualifiers::ATOMIC)
        } else {
            None
        }
    }

    fn read_base_type(&mut self) -> Option<QualifiableType> {
        if let Some(prim) = self.read_primitive_type() {
            return Some(QualifiableType::Prim(prim));
        };
        match self.iter.peek() {
            Some(PositionedToken(Token::Identifier(identifier), _)) => {
                // TODO: Clone should not be needed (moving is_type_name to another struct might do the trick)
                let copy = identifier.clone();
                if self.is_type_name(copy.as_ref()) {
                    panic!("TODO")
                } else {
                    None
                }
            }
            Some(PositionedToken(Token::Keyword(Keyword::Int), _)) => {
                self.iter.next();
                Some(QualifiableType::Prim(PrimitiveType::Int))
            }
            _ => None,
        }
    }

    fn read_primitive_type(&mut self) -> Option<PrimitiveType> {
        if self.iter.advance_if_kw(Keyword::Void) {
            Some(PrimitiveType::Void)
        } else if self.iter.advance_if_kw(Keyword::Char) {
            Some(PrimitiveType::Char)
        } else if self.iter.advance_if_kw(Keyword::Signed) {
            if self.iter.advance_if_kw(Keyword::Char) {
                Some(PrimitiveType::SignedChar)
            } else if self.iter.advance_if_kw(Keyword::Short) {
                self.iter.advance_if_kw(Keyword::Int);
                Some(PrimitiveType::Short)
            } else if self.iter.advance_if_kw(Keyword::Long) {
                if self.iter.advance_if_kw(Keyword::Long) {
                    self.iter.advance_if_kw(Keyword::Int);
                    Some(PrimitiveType::LongLong)
                } else {
                    self.iter.advance_if_kw(Keyword::Int);
                    Some(PrimitiveType::Long)
                }
            } else {
                self.iter.advance_if_kw(Keyword::Int);
                Some(PrimitiveType::Int)
            }
        } else if self.iter.advance_if_kw(Keyword::Unsigned) {
            if self.iter.advance_if_kw(Keyword::Char) {
                Some(PrimitiveType::UnsignedChar)
            } else if self.iter.advance_if_kw(Keyword::Short) {
                self.iter.advance_if_kw(Keyword::Int);
                Some(PrimitiveType::UnsignedShort)
            } else if self.iter.advance_if_kw(Keyword::Long) {
                if self.iter.advance_if_kw(Keyword::Long) {
                    self.iter.advance_if_kw(Keyword::Int);
                    Some(PrimitiveType::UnsignedLongLong)
                } else {
                    self.iter.advance_if_kw(Keyword::Int);
                    Some(PrimitiveType::UnsignedLong)
                }
            } else {
                self.iter.advance_if_kw(Keyword::Int);
                Some(PrimitiveType::UnsignedInt)
            }
        } else if self.iter.advance_if_kw(Keyword::Short) {
            self.iter.advance_if_kw(Keyword::Int);
            Some(PrimitiveType::Short)
        } else if self.iter.advance_if_kw(Keyword::Int) {
            Some(PrimitiveType::Int)
        } else if self.iter.advance_if_kw(Keyword::Long) {
            if self.iter.advance_if_kw(Keyword::Long) {
                self.iter.advance_if_kw(Keyword::Int);
                Some(PrimitiveType::LongLong)
            } else if self.iter.advance_if_kw(Keyword::Double) {
                if self.iter.advance_if_kw(Keyword::Complex) {
                    Some(PrimitiveType::LongDoubleComplex)
                } else {
                    Some(PrimitiveType::LongDouble)
                }
            } else {
                self.iter.advance_if_kw(Keyword::Int);
                Some(PrimitiveType::Long)
            }
        } else if self.iter.advance_if_kw(Keyword::Float) {
            if self.iter.advance_if_kw(Keyword::Complex) {
                Some(PrimitiveType::FloatComplex)
            } else {
                Some(PrimitiveType::Float)
            }
        } else if self.iter.advance_if_kw(Keyword::Double) {
            if self.iter.advance_if_kw(Keyword::Complex) {
                Some(PrimitiveType::DoubleComplex)
            } else {
                Some(PrimitiveType::Double)
            }
        } else if self.iter.advance_if_kw(Keyword::Bool) {
            Some(PrimitiveType::Bool)
        } else {
            None
        }
    }

    fn expect_token(&mut self, expected_token: Token) -> Result<(), ParseError> {
        match self.iter.next() {
            Some(PositionedToken(token, position)) => if token == expected_token {
                Ok(())
            } else {
                Err(ParseError::UnexpectedToken(token, position))
            },
            None => Err(ParseError::ExpectingToken(expected_token)),
        }
    }

    // Should be called just after having read an opening parenthesis.
    fn read_func_args(&mut self) -> Result<FunctionParameters, ParseError> {
        if self.iter.advance_if_punc(Punctuator::RightParenthesis) {
            return Ok(FunctionParameters::Undefined);
        }
        // TODO: Handle parameters
        panic!("TODO: Handle parameters")
    }

    fn read_next_external_declaration(&mut self) -> Result<Option<ExternalDecl>, ParseError> {
        if self.iter.peek().is_none() {
            return Ok(None);
        }
        let base_qual_ty = {
            let mut qualifiers = TypeQualifiers::empty();
            while let Some(qualifier) = self.read_type_qualifier() {
                qualifiers |= qualifier;
            }
            let ty = self
                .read_base_type()
                .unwrap_or(QualifiableType::Prim(PrimitiveType::Int));
            while let Some(qualifier) = self.read_type_qualifier() {
                qualifiers |= qualifier;
            }
            QualifiedType(ty, qualifiers)
        };
        let mut ptr_qualifs = Vec::new();
        let mut ptr_qualifs_stack = Vec::new();
        let ident = loop {
            if self.iter.advance_if_punc(Punctuator::Star) {
                let mut qualifiers = TypeQualifiers::empty();
                while let Some(qualifier) = self.read_type_qualifier() {
                    qualifiers |= qualifier;
                }
                ptr_qualifs.push(qualifiers);
            } else if self.iter.advance_if_punc(Punctuator::LeftParenthesis) {
                ptr_qualifs_stack.push(ptr_qualifs);
                ptr_qualifs = Vec::new();
            } else if let Some(ident) = self.iter.next_if_any_ident() {
                ptr_qualifs_stack.push(ptr_qualifs);
                break ident;
            } else if self.iter.advance_if_punc(Punctuator::Semicolon) {
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
            if self.iter.advance_if_punc(Punctuator::LeftParenthesis) {
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

        let mut def_ty = DefinableType::Qual(base_qual_ty);
        while let Some((ptr_qualifs, func_args)) = ptr_qualifs_reversed_stack.pop() {
            for qualifiers in ptr_qualifs {
                def_ty = DefinableType::Qual(QualifiedType(
                    QualifiableType::Ptr(Box::new(def_ty)),
                    qualifiers,
                ));
            }
            if let Some(func_args) = func_args {
                def_ty = match def_ty {
                    DefinableType::Qual(qual_ty) => {
                        DefinableType::Func(FunctionType(qual_ty, func_args))
                    }
                    DefinableType::Array(_) | DefinableType::Func(_) => {
                        panic!("you can't return an array or func - TODO: proper error")
                    }
                };
            }
        }

        Ok(Some(ExternalDecl::Decl(ident, def_ty)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_one_external_declaration(code: &str) -> Option<ExternalDecl> {
        let mut parser = match Parser::from_code(code) {
            Ok(parser) => parser,
            Err(err) => panic!(r#"Unexpected lexer error {:?} for "{:}""#, err, code),
        };
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
        assert_eq!(
            parse_one_external_declaration(r#"signed long;"#),
            Some(ExternalDecl::Nothing)
        );
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
}

fn main() -> Result<(), ParseError> {
    let mut parser = Parser::from_code(r#"x;"#)?;
    let decl = parser.read_next_external_declaration()?;
    println!("Declaration: {:?}", decl);
    Ok(())
}
