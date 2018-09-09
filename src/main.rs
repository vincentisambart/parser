// Notes
// - For testing pragmas, have a look at clang's test/Sema/pragma-align-packed.c
mod lex;
mod scan;
use crate::lex::{Keyword, LexError, Position, Punctuator, Token, TokenIter};
use crate::scan::{Scan, ScanErr};
use std::iter::Peekable;

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

struct Parser<'a> {
    iter: Peekable<TokenIter<'a>>,
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

impl<'a> Parser<'a> {
    fn from(code: &'a str) -> Parser<'a> {
        let iter = TokenIter::from(code).peekable();
        Parser { iter }
    }

    fn is_type_name(&self, _name: &str) -> bool {
        false // TODO
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

    fn scan_kw(&mut self, kw: Keyword) -> Result<bool, LexError> {
        let token = self.iter.scan_one_err(|token| match token.token() {
            Token::Keyword(x) if x == kw => true,
            _ => false,
        })?;
        Ok(token.is_some())
    }

    fn scan_punc(&mut self, punc: Punctuator) -> Result<bool, LexError> {
        let token = self.iter.scan_one_err(|token| match token.token() {
            Token::Punctuator(x) if x == punc => true,
            _ => false,
        })?;
        Ok(token.is_some())
    }

    fn read_primitive_type(&mut self) -> Result<Option<PrimitiveType>, ParseError> {
        let ty = if self.scan_kw(Keyword::Void)? {
            Some(PrimitiveType::Void)
        } else if self.scan_kw(Keyword::Char)? {
            Some(PrimitiveType::Char)
        } else if self.scan_kw(Keyword::Signed)? {
            if self.scan_kw(Keyword::Char)? {
                Some(PrimitiveType::SignedChar)
            } else if self.scan_kw(Keyword::Short)? {
                self.scan_kw(Keyword::Int)?;
                Some(PrimitiveType::Short)
            } else if self.scan_kw(Keyword::Long)? {
                if self.scan_kw(Keyword::Long)? {
                    self.scan_kw(Keyword::Int)?;
                    Some(PrimitiveType::LongLong)
                } else {
                    self.scan_kw(Keyword::Int)?;
                    Some(PrimitiveType::Long)
                }
            } else {
                self.scan_kw(Keyword::Int)?;
                Some(PrimitiveType::Int)
            }
        } else if self.scan_kw(Keyword::Unsigned)? {
            if self.scan_kw(Keyword::Char)? {
                Some(PrimitiveType::UnsignedChar)
            } else if self.scan_kw(Keyword::Short)? {
                self.scan_kw(Keyword::Int)?;
                Some(PrimitiveType::UnsignedShort)
            } else if self.scan_kw(Keyword::Long)? {
                if self.scan_kw(Keyword::Long)? {
                    self.scan_kw(Keyword::Int)?;
                    Some(PrimitiveType::UnsignedLongLong)
                } else {
                    self.scan_kw(Keyword::Int)?;
                    Some(PrimitiveType::UnsignedLong)
                }
            } else {
                self.scan_kw(Keyword::Int)?;
                Some(PrimitiveType::UnsignedInt)
            }
        } else if self.scan_kw(Keyword::Short)? {
            self.scan_kw(Keyword::Int)?;
            Some(PrimitiveType::Short)
        } else if self.scan_kw(Keyword::Int)? {
            Some(PrimitiveType::Int)
        } else if self.scan_kw(Keyword::Long)? {
            if self.scan_kw(Keyword::Long)? {
                self.scan_kw(Keyword::Int)?;
                Some(PrimitiveType::LongLong)
            } else if self.scan_kw(Keyword::Double)? {
                if self.scan_kw(Keyword::Complex)? {
                    Some(PrimitiveType::LongDoubleComplex)
                } else {
                    Some(PrimitiveType::LongDouble)
                }
            } else {
                self.scan_kw(Keyword::Int)?;
                Some(PrimitiveType::Long)
            }
        } else if self.scan_kw(Keyword::Float)? {
            if self.scan_kw(Keyword::Complex)? {
                Some(PrimitiveType::FloatComplex)
            } else {
                Some(PrimitiveType::Float)
            }
        } else if self.scan_kw(Keyword::Double)? {
            if self.scan_kw(Keyword::Complex)? {
                Some(PrimitiveType::DoubleComplex)
            } else {
                Some(PrimitiveType::Double)
            }
        } else if self.scan_kw(Keyword::Bool)? {
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
        while self.scan_punc(Punctuator::Star)? {
            ty = Type::Pointer(Box::new(ty));
        }
        let token = match self.iter.peek() {
            Some(Ok(token)) => token,
            Some(Err(err)) => return Err(err.clone().into()),
            None => {
                return Err(ParseError::ExpectingToken(Token::Punctuator(
                    Punctuator::Semicolon,
                )))
            }
        };
        let identifier = match token.token() {
            Token::Identifier(_) => {
                // TODO: Check if identifier not typedef
                if let Token::Identifier(identifier) = self.iter.next().unwrap().unwrap().token() {
                    identifier
                } else {
                    unreachable!()
                }
            }
            Token::Punctuator(Punctuator::Semicolon) => {
                self.iter.next();
                return Ok(Some(ExternalDeclaration::Nothing));
            }
            _ => return Err(ParseError::UnexpectedToken(token.token(), token.position())),
        };

        let token = match self.iter.peek() {
            Some(Ok(token)) => token,
            Some(Err(err)) => return Err(err.clone().into()),
            None => {
                return Err(ParseError::ExpectingToken(Token::Punctuator(
                    Punctuator::Semicolon,
                )))
            }
        };
        match token.token() {
            Token::Punctuator(Punctuator::Semicolon) => {
                self.iter.next();
                Ok(Some(ExternalDeclaration::Declaration(identifier, ty)))
            }
            _ => Err(ParseError::UnexpectedToken(token.token(), token.position())),
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
