// Notes
// - For testing pragmas, have a look at clang's test/Sema/pragma-align-packed.c
mod lex;
mod scan;
use crate::lex::{Keyword, LexError, Position, Punctuator, Token, TokenIter};
use std::iter::Peekable;

#[derive(Debug, Clone, PartialEq)]
enum Type {
    Int,
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

    fn read_type(&mut self) -> Result<Option<Type>, LexError> {
        // read_type should not be called when no next token is available
        let token = match self.iter.peek() {
            Some(Ok(token)) => token,
            Some(Err(err)) => return Err(err.clone()),
            None => panic!("read_type should not be called when no next token is available"),
        };
        match token.token() {
            Token::Identifier(identifier) => if self.is_type_name(identifier.as_ref()) {
                panic!("TODO")
            } else {
                Ok(None)
            },
            Token::Keyword(Keyword::Int) => {
                self.iter.next();
                Ok(Some(Type::Int))
            }
            _ => Ok(None),
        }
    }

    fn read_next_external_declaration(
        &mut self,
    ) -> Result<Option<ExternalDeclaration>, ParseError> {
        if self.iter.peek().is_none() {
            return Ok(None);
        }
        let ty = self.read_type()?.unwrap_or(Type::Int);
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
            _ => return Err(ParseError::UnexpectedToken(token.token(), token.position())),
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
                Type::Int,
            )),
        );
        assert_eq!(
            parse_one_external_declaration(r#"int abcd;"#),
            Some(ExternalDeclaration::Declaration(
                "abcd".to_string(),
                Type::Int,
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
