use crate::lex::{Position, Token};

#[derive(Debug, Clone)]
pub enum ParseErrorKind {
    UnexpectedEOF,
    UnexpectedToken(Token),
    UnexpectedChar(char),
    InvalidPreprocDirective,
    InvalidConstruct,
}

#[derive(Debug, Clone)]
pub struct ParseError {
    position: Option<Position>,
    kind: ParseErrorKind,
    message: String,
}

impl ParseError {
    pub fn new(kind: ParseErrorKind, message: String) -> ParseError {
        ParseError {
            position: None,
            kind,
            message,
        }
    }

    pub fn new_with_position(
        kind: ParseErrorKind,
        message: String,
        position: Position,
    ) -> ParseError {
        ParseError {
            position: Some(position),
            kind,
            message,
        }
    }
}
