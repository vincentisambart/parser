// Notes
// - For testing pragmas, have a look at clang's test/Sema/pragma-align-packed.c
mod lex;
mod scan;
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

#[derive(Debug, Clone, PartialEq)]
enum Type {
    Primitive(PrimitiveType),
    Pointer(Box<Type>),
    Custom(String, Rc<Type>),
}

#[derive(Debug, Clone, PartialEq)]
enum FunctionArguments {
    Undefined,
    Defined {
        args: Vec<(String, Type)>,
        var_args: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
struct FunctionType {
    return_type: Type,
    arguments: FunctionArguments,
}

#[derive(Debug, Clone, PartialEq)]
enum ExternalDeclaration {
    Declaration(String, Type),
    FunctionDefinition(String, FunctionType),
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

    fn read_type(&mut self) -> Result<Option<Type>, ParseError> {
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

    fn read_next_external_declaration(
        &mut self,
    ) -> Result<Option<ExternalDeclaration>, ParseError> {
        if self.iter.peek().is_none() {
            return Ok(None);
        }
        let mut ty = self
            .read_type()?
            .unwrap_or(Type::Primitive(PrimitiveType::Int));
        while self.iter.advance_if_punc(Punctuator::Star)? {
            // TODO: Should be in read_type?
            ty = Type::Pointer(Box::new(ty));
        }
        if self.iter.advance_if_punc(Punctuator::Semicolon)? {
            return Ok(Some(ExternalDeclaration::Nothing));
        }
        let ident = if let Some(ident) = self.iter.next_if_any_ident()? {
            ident
        } else {
            return match self.iter.next() {
                Some(Ok(token)) => {
                    Err(ParseError::UnexpectedToken(token.token(), token.position()))
                }
                Some(Err(err)) => Err(err.into()),
                None => Err(ParseError::ExpectingToken(Token::Punctuator(
                    Punctuator::Semicolon,
                ))),
            };
        };

        if self.iter.advance_if_punc(Punctuator::Semicolon)? {
            Ok(Some(ExternalDeclaration::Declaration(ident, ty)))
        } else {
            match self.iter.next() {
                Some(Ok(token)) => {
                    Err(ParseError::UnexpectedToken(token.token(), token.position()))
                }
                Some(Err(err)) => return Err(err.into()),
                None => {
                    return Err(ParseError::ExpectingToken(Token::Punctuator(
                        Punctuator::Semicolon,
                    )))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_one_external_declaration(code: &str) -> Option<ExternalDeclaration> {
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
            Some(ExternalDeclaration::Declaration(
                "abcd".to_string(),
                Type::Primitive(PrimitiveType::Int),
            )),
        );
        // assert_eq!(
        //     parse_one_external_declaration(r#"(abcd);"#),
        //     Some(ExternalDeclaration::Declaration(
        //         "abcd".to_string(),
        //         Type::Primitive(PrimitiveType::Int),
        //     )),
        // );
        assert_eq!(
            parse_one_external_declaration(r#"int abcd;"#),
            Some(ExternalDeclaration::Declaration(
                "abcd".to_string(),
                Type::Primitive(PrimitiveType::Int),
            )),
        );
        assert_eq!(
            parse_one_external_declaration(r#"unsigned char abcd;"#),
            Some(ExternalDeclaration::Declaration(
                "abcd".to_string(),
                Type::Primitive(PrimitiveType::UnsignedChar),
            )),
        );
        assert_eq!(
            parse_one_external_declaration(r#"signed long long int * abcd;"#),
            Some(ExternalDeclaration::Declaration(
                "abcd".to_string(),
                Type::Pointer(Box::new(Type::Primitive(PrimitiveType::LongLong))),
            )),
        );
    }
}

fn main() -> Result<(), LexError> {
    let mut parser = Parser::from(r#"x;"#);
    let decl = parser.read_next_external_declaration();
    println!("Declaration: {:?}", decl);
    Ok(())
}
