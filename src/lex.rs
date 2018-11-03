// TODO:
// - maybe use Rc for strings in tokens
// - maybe support universal character names
// - maybe support other non-ascii identifiers (check what clang does)
// - handle more preprocessing directives (#error, unknown #pragma should be an error...)

use crate::error::{ParseError, ParseErrorKind};
use crate::failable::{FailableIterator, FailablePeekable};
use crate::peeking::Peeking;

use lazy_static::lazy_static;
use std::char;
use std::collections::HashMap;
use std::iter::Peekable;
use std::rc::Rc;
use std::str::Chars;

macro_rules! any_of {
    ( $( $x:pat ),* ) => {
        |c| match c {
            $(
                $x => true,
            )*
            _ => false,
        }
    };
}

trait CharPattern {
    fn is_ascii_octdigit(&self) -> bool;
}

impl CharPattern for char {
    fn is_ascii_octdigit(&self) -> bool {
        match self {
            '0'..='7' => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerRepr {
    Dec,
    Hex,
    Oct,
}

impl IntegerRepr {
    pub fn radix(self) -> u32 {
        match self {
            IntegerRepr::Dec => 10,
            IntegerRepr::Hex => 16,
            IntegerRepr::Oct => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatRepr {
    Dec,
    Hex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerSuffix {
    Unsigned,
    Long,
    LongLong,
    UnsignedLong,
    UnsignedLongLong,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatSuffix {
    Float,
    LongDouble,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharPrefix {
    WChar,
    Char16,
    Char32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringPrefix {
    Utf8,
    WChar,
    Char16,
    Char32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Punctuator {
    LeftSquareBracket,
    RightSquareBracket,
    LeftParenthesis,
    RightParenthesis,
    LeftCurlyBracket,
    RightCurlyBracket,
    Period,
    Arrow,
    PlusPlus,
    MinusMinus,
    Ampersand,
    Star,
    Plus,
    Minus,
    Tilde,
    Exclamation,
    Slash,
    Percent,
    LessLess,
    GreaterGreater,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    EqualEqual,
    ExclamationEqual,
    Caret,
    Pipe,
    AmpersandAmpersand,
    PipePipe,
    Question,
    Colon,
    Semicolon,
    Ellipsis,
    Equal,
    StarEqual,
    SlashEqual,
    PercentEqual,
    PlusEqual,
    MinusEqual,
    LessLessEqual,
    GreaterGreaterEqual,
    AmpersandEqual,
    CaretEqual,
    PipeEqual,
    Comma,
    Hash,
    HashHash,
    At,
}

#[derive(Debug, Clone)]
pub struct Position {
    file_path: Rc<String>,
    line: u32,
}

macro_rules! kw_enum {
    ( $name:ident { $( $kw:ident => $repr:expr, )* } ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name {
            $($kw),*
        }

        impl $name {
            // pub fn to_str(&self) -> &'static str {
            //     match self {
            //         $($name::$kw => $repr,)*
            //     }
            // }

            fn map() -> HashMap<&'static str, $name> {
                let mut keywords = HashMap::new();
                $(keywords.insert($repr, $name::$kw);)*
                keywords
            }
        }
    }
}

kw_enum! {
    Keyword {
        Auto => "auto",
        Break => "break",
        Case => "case",
        Char => "char",
        Const => "const",
        Continue => "continue",
        Default => "default",
        Do => "do",
        Double => "double",
        Else => "else",
        Enum => "enum",
        Extern => "extern",
        Float => "float",
        For => "for",
        Goto => "goto",
        If => "if",
        Inline => "inline",
        InlineAlt => "__inline",
        Int => "int",
        Long => "long",
        Register => "register",
        Restrict => "restrict",
        Return => "return",
        Short => "short",
        Signed => "signed",
        Sizeof => "sizeof",
        Static => "static",
        Struct => "struct",
        Switch => "switch",
        Typedef => "typedef",
        Union => "union",
        Unsigned => "unsigned",
        Void => "void",
        Volatile => "volatile",
        While => "while",
        Alignas => "_Alignas",
        Alignof => "_Alignof",
        Atomic => "_Atomic",
        Bool => "_Bool",
        Complex => "_Complex",
        Generic => "_Generic",
        Imaginary => "_Imaginary",
        Noreturn => "_Noreturn",
        StaticAssert => "_Static_assert",
        ThreadLocal => "_Thread_local",
    }
}

// Objective-C keywords, always after a @.
// The list comes from clang's include/clang/Basic/TokenKinds.def.
kw_enum! {
    ObjCKeyword {
        Class => "class",
        CompatibilityAlias => "compatibility_alias",
        Defs => "defs",
        Encode => "encode",
        End => "end",
        Implementation => "implementation",
        Interface => "interface",
        Private => "private",
        Protected => "protected",
        Protocol => "protocol",
        Public => "public",
        Selector => "selector",
        Throw => "throw",
        Try => "try",
        Catch => "catch",
        Finally => "finally",
        Synchronized => "synchronized",
        Autoreleasepool => "autoreleasepool",
        Property => "property",
        Package => "package",
        Required => "required",
        Optional => "optional",
        Synthesize => "synthesize",
        Dynamic => "dynamic",
        Import => "import",
        Available => "available",
    }
}

lazy_static! {
    static ref KEYWORDS: HashMap<&'static str, Keyword> = Keyword::map();
    static ref OBJC_KEYWORDS: HashMap<&'static str, ObjCKeyword> = ObjCKeyword::map();
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    Integer(String, IntegerRepr, Option<IntegerSuffix>),
    Float(String, FloatRepr, Option<FloatSuffix>),
    Char(char, Option<CharPrefix>),
    Str(String, Option<StringPrefix>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Literal(Literal),
    Identifier(String),
    Keyword(Keyword),
    ObjCKeyword(ObjCKeyword),
    Punctuator(Punctuator),
}

#[derive(Debug, Clone)]
pub struct PositionedToken(pub Token, pub Position);

struct BasicTokenIter<'a> {
    scanner: Peekable<Chars<'a>>,
    position: Position,
    is_start_of_line: bool,
    previous_token: Option<Token>,
    next_token_to_return: Option<Token>,
}

impl<'a> BasicTokenIter<'a> {
    fn from(code: &'a str) -> BasicTokenIter<'a> {
        let scanner = code.chars().peekable();
        let position = Position {
            file_path: Rc::new("<unknown>".to_string()),
            line: 1,
        };
        BasicTokenIter {
            scanner,
            position,
            is_start_of_line: true,
            previous_token: None,
            next_token_to_return: None,
        }
    }

    fn read_number_literal(&mut self) -> Result<Token, ParseError> {
        let mut token = String::new();

        if self.scanner.advance_if(any_of!('0')) {
            if self.scanner.advance_if(any_of!('x', 'X')) {
                // hexadecimal
                self.scanner.push_while(&mut token, char::is_ascii_hexdigit);
                assert!(!token.is_empty());
                let mut is_float = false;
                if self.scanner.advance_if(any_of!('.')) {
                    is_float = true;
                    token.push('.');
                    self.scanner.push_while(&mut token, char::is_ascii_hexdigit);
                }
                if let Some(p) = self.scanner.next_if(any_of!('p', 'P')) {
                    is_float = true;
                    token.push(p);
                    if let Some(sign) = self.scanner.next_if(any_of!('-', '+')) {
                        token.push(sign);
                    }
                    self.scanner.push_while(&mut token, char::is_ascii_digit);
                }
                if is_float {
                    let suffix = self.read_float_suffix();
                    return Ok(Token::Literal(Literal::Float(
                        token,
                        FloatRepr::Hex,
                        suffix,
                    )));
                } else {
                    let suffix = self.read_integer_suffix();
                    return Ok(Token::Literal(Literal::Integer(
                        token,
                        IntegerRepr::Hex,
                        suffix,
                    )));
                }
            } else if let Some(digit) = self.scanner.next_if(CharPattern::is_ascii_octdigit) {
                // octal
                token.push(digit);
                self.scanner
                    .push_while(&mut token, CharPattern::is_ascii_octdigit);
                return Ok(Token::Literal(Literal::Integer(
                    token,
                    IntegerRepr::Oct,
                    self.read_integer_suffix(),
                )));
            }

            token.push('0');
        }

        self.scanner.push_while(&mut token, char::is_ascii_digit);
        let mut is_float = false;
        if self.scanner.advance_if(any_of!('.')) {
            is_float = true;
            token.push('.');
            self.scanner.push_while(&mut token, char::is_ascii_digit);
            if let Some(e) = self.scanner.next_if(any_of!('e', 'E')) {
                token.push(e);
                if let Some(sign) = self.scanner.next_if(any_of!('-', '+')) {
                    token.push(sign);
                }
                self.scanner.push_while(&mut token, char::is_ascii_digit);
            }
        }
        if let Some(e) = self.scanner.next_if(any_of!('e', 'E')) {
            is_float = true;
            token.push(e);
            if let Some(sign) = self.scanner.next_if(any_of!('-', '+')) {
                token.push(sign);
            }
            self.scanner.push_while(&mut token, char::is_ascii_digit);
        }
        if is_float {
            let suffix = self.read_float_suffix();
            Ok(Token::Literal(Literal::Float(
                token,
                FloatRepr::Dec,
                suffix,
            )))
        } else {
            let suffix = self.read_integer_suffix();
            Ok(Token::Literal(Literal::Integer(
                token,
                IntegerRepr::Dec,
                suffix,
            )))
        }
    }

    fn read_integer_suffix(&mut self) -> Option<IntegerSuffix> {
        if self.scanner.advance_if(any_of!('l', 'L')) {
            if self.scanner.advance_if(any_of!('l', 'L')) {
                Some(IntegerSuffix::LongLong)
            } else {
                Some(IntegerSuffix::Long)
            }
        } else if self.scanner.advance_if(any_of!('u', 'U')) {
            if self.scanner.advance_if(any_of!('l', 'L')) {
                if self.scanner.advance_if(any_of!('l', 'L')) {
                    Some(IntegerSuffix::UnsignedLongLong)
                } else {
                    Some(IntegerSuffix::UnsignedLong)
                }
            } else {
                Some(IntegerSuffix::Unsigned)
            }
        } else {
            None
        }
    }

    fn read_float_suffix(&mut self) -> Option<FloatSuffix> {
        if self.scanner.advance_if(any_of!('f', 'F')) {
            Some(FloatSuffix::Float)
        } else if self.scanner.advance_if(any_of!('l', 'L')) {
            Some(FloatSuffix::LongDouble)
        } else {
            None
        }
    }

    // The list of escapes comes from clang's LiteralSupport.cpp
    fn read_char_escape(&mut self) -> Result<char, ParseError> {
        match self.scanner.next() {
            None => Err(ParseError::new(
                ParseErrorKind::UnexpectedEOF,
                "unfinished character escape at end of file".to_string(),
            )),
            Some('a') => Ok('\x07'),             // bell
            Some('b') => Ok('\x08'),             // backspace
            Some('e') | Some('E') => Ok('\x1b'), // escape
            Some('f') => Ok('\x0c'),             // form feed
            Some('n') => Ok('\n'),               // line feed
            Some('r') => Ok('\r'),               // carriage return
            Some('t') => Ok('\t'),               // tab
            Some('v') => Ok('\x0b'),             // vertical tab
            // hexadecimal escape
            Some('x') => {
                let mut x = 0u32;
                while let Some(c) = self.scanner.next_if(char::is_ascii_hexdigit) {
                    let digit = c.to_digit(16).unwrap();
                    x = (x << 4) | digit;
                }
                Ok(char::from_u32(x).unwrap())
            }
            // octal escape
            Some(c) if c.is_ascii_octdigit() => {
                let mut x: u32 = c.to_digit(8).unwrap();
                let mut len = 1;
                while let Some(c) = self.scanner.next_if(CharPattern::is_ascii_octdigit) {
                    let digit = c.to_digit(8).unwrap();
                    x = (x << 3) | digit;
                    // an octal escape is 3 chars max
                    len += 1;
                    if len >= 3 {
                        break;
                    }
                }
                Ok(char::from_u32(x).unwrap())
            }
            Some(c) => Ok(c),
        }
    }

    fn char_literal_prefix(prefix: &str) -> Option<CharPrefix> {
        match prefix {
            "L" => Some(CharPrefix::WChar),
            "u" => Some(CharPrefix::Char16),
            "U" => Some(CharPrefix::Char32),
            _ => None,
        }
    }

    fn read_char_literal(&mut self, prefix: Option<CharPrefix>) -> Result<Token, ParseError> {
        self.scanner
            .next_if(any_of!('\''))
            .expect("read_char_literal should only be called when next character is \"'\"");
        let t = match self.scanner.next() {
            Some('\\') => {
                let c = self.read_char_escape()?;
                Token::Literal(Literal::Char(c, prefix))
            }
            Some(c) => Token::Literal(Literal::Char(c, prefix)),
            None => {
                return Err(ParseError::new(
                    ParseErrorKind::UnexpectedEOF,
                    "unfinished character literal at the end of the file".to_string(),
                ))
            }
        };
        match self.scanner.next() {
            Some('\'') => Ok(t),
            // TODO: We might want to allow multi-characters constants (at least FourCCs)
            // https://stackoverflow.com/a/47312044
            Some(c) => Err(ParseError::new_with_position(
                ParseErrorKind::UnexpectedChar(c),
                "a character literal should only have one character".to_string(),
                self.position.clone(),
            )),
            None => Err(ParseError::new(
                ParseErrorKind::UnexpectedEOF,
                "unfinished character literal at the end of the file".to_string(),
            )),
        }
    }

    fn string_literal_prefix(prefix: &str) -> Option<StringPrefix> {
        match prefix {
            "L" => Some(StringPrefix::WChar),
            "u8" => Some(StringPrefix::Utf8),
            "u" => Some(StringPrefix::Char16),
            "U" => Some(StringPrefix::Char32),
            _ => None,
        }
    }

    fn read_string_literal(&mut self, prefix: Option<StringPrefix>) -> Result<Token, ParseError> {
        self.scanner
            .next_if(any_of!('"'))
            .expect("read_string_literal should only be called when next character is '\"'");

        let mut string = String::new();

        loop {
            match self.scanner.next() {
                Some('"') => break Ok(Token::Literal(Literal::Str(string, prefix))),
                Some('\\') => {
                    let c = self.read_char_escape()?;
                    string.push(c);
                }
                Some(c) => string.push(c),
                None => {
                    break Err(ParseError::new(
                        ParseErrorKind::UnexpectedEOF,
                        "unfinished string literal at the end of the file".to_string(),
                    ))
                }
            }
        }
    }

    fn handle_preprocessing_directive(&mut self, directive: &[Token]) -> Result<(), ParseError> {
        let mut iter = directive.iter();
        match iter.next() {
            None => (), // empty preprocessing directive
            Some(Token::Identifier(identifier)) => match identifier.as_str() {
                // Handle "#line 1 file" as "#1 file"
                "line" => return self.handle_preprocessing_directive(&directive[1..]),
                _ => {
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidPreprocDirective,
                        "#line is currently the only handled preprocessing directive".to_string(),
                        self.position.clone(),
                    ))
                }
            },
            Some(Token::Literal(Literal::Integer(literal, repr, _))) => {
                let line = if let Ok(number) = u32::from_str_radix(literal.as_ref(), repr.radix()) {
                    number
                } else {
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidPreprocDirective,
                        "invalid line number in line number preprocessing directive".to_string(),
                        self.position.clone(),
                    ));
                };

                let file_path;
                if let Some(Token::Literal(Literal::Str(text, None))) = iter.next() {
                    file_path = text;
                } else {
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidPreprocDirective,
                        "invalid file name in line number preprocessing directive".to_string(),
                        self.position.clone(),
                    ));
                }
                self.position.line = line;
                self.position.file_path = Rc::new(file_path.clone());

                for token in iter {
                    match token {
                        Token::Literal(Literal::Integer(_, _, _)) => (),
                        _ => {
                            return Err(ParseError::new_with_position(
                                ParseErrorKind::InvalidPreprocDirective,
                                "a line number preprocessing directive should end with numbers"
                                    .to_string(),
                                self.position.clone(),
                            ))
                        }
                    }
                }
            }
            _ => {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidPreprocDirective,
                    "a preprocessing directive should start with a number or an identifier"
                        .to_string(),
                    self.position.clone(),
                ))
            }
        }
        Ok(())
    }

    fn read_punctuator(&mut self) -> Result<Token, ParseError> {
        let punc = match self
            .scanner
            .next()
            .expect("read_punctuator should be called with at least one char available")
        {
            '[' => Punctuator::LeftSquareBracket,
            ']' => Punctuator::RightSquareBracket,
            '(' => Punctuator::LeftParenthesis,
            ')' => Punctuator::RightParenthesis,
            '{' => Punctuator::LeftCurlyBracket,
            '}' => Punctuator::RightCurlyBracket,
            '-' => {
                if self.scanner.advance_if(any_of!('-')) {
                    Punctuator::MinusMinus
                } else if self.scanner.advance_if(any_of!('>')) {
                    Punctuator::Arrow
                } else if self.scanner.advance_if(any_of!('=')) {
                    Punctuator::MinusEqual
                } else {
                    Punctuator::Minus
                }
            }
            '+' => {
                if self.scanner.advance_if(any_of!('+')) {
                    Punctuator::PlusPlus
                } else if self.scanner.advance_if(any_of!('=')) {
                    Punctuator::PlusEqual
                } else {
                    Punctuator::Plus
                }
            }
            '&' => {
                if self.scanner.advance_if(any_of!('&')) {
                    Punctuator::AmpersandAmpersand
                } else if self.scanner.advance_if(any_of!('=')) {
                    Punctuator::AmpersandEqual
                } else {
                    Punctuator::Ampersand
                }
            }
            '|' => {
                if self.scanner.advance_if(any_of!('|')) {
                    Punctuator::PipePipe
                } else if self.scanner.advance_if(any_of!('=')) {
                    Punctuator::PipeEqual
                } else {
                    Punctuator::Pipe
                }
            }
            '<' => {
                if self.scanner.advance_if(any_of!('<')) {
                    if self.scanner.advance_if(any_of!('=')) {
                        Punctuator::LessLessEqual
                    } else {
                        Punctuator::LessLess
                    }
                } else if self.scanner.advance_if(any_of!('=')) {
                    Punctuator::LessEqual
                } else if self.scanner.advance_if(any_of!(':')) {
                    // digraph
                    Punctuator::LeftSquareBracket
                } else if self.scanner.advance_if(any_of!('%')) {
                    // digraph
                    Punctuator::LeftCurlyBracket
                } else {
                    Punctuator::Less
                }
            }
            '>' => {
                if self.scanner.advance_if(any_of!('>')) {
                    if self.scanner.advance_if(any_of!('=')) {
                        Punctuator::GreaterGreaterEqual
                    } else {
                        Punctuator::GreaterGreater
                    }
                } else if self.scanner.advance_if(any_of!('=')) {
                    Punctuator::GreaterEqual
                } else {
                    Punctuator::Greater
                }
            }
            '.' => {
                if self.scanner.advance_if(any_of!('.')) {
                    if self.scanner.advance_if(any_of!('.')) {
                        Punctuator::Ellipsis
                    } else {
                        self.next_token_to_return = Some(Token::Punctuator(Punctuator::Period));
                        Punctuator::Period
                    }
                } else {
                    Punctuator::Period
                }
            }
            '*' => {
                if self.scanner.advance_if(any_of!('=')) {
                    Punctuator::StarEqual
                } else {
                    Punctuator::Star
                }
            }
            '~' => Punctuator::Tilde,
            '!' => {
                if self.scanner.advance_if(any_of!('=')) {
                    Punctuator::ExclamationEqual
                } else {
                    Punctuator::Exclamation
                }
            }
            '/' => {
                if self.scanner.advance_if(any_of!('=')) {
                    Punctuator::SlashEqual
                } else {
                    Punctuator::Slash
                }
            }
            '%' => {
                if self.scanner.advance_if(any_of!('=')) {
                    Punctuator::PercentEqual
                } else if self.scanner.advance_if(any_of!('>')) {
                    // digraph
                    Punctuator::RightCurlyBracket
                } else if self.scanner.advance_if(any_of!(':')) {
                    // digraph
                    if self.scanner.advance_if(any_of!('%')) {
                        if self.scanner.advance_if(any_of!(':')) {
                            Punctuator::HashHash
                        } else {
                            if self.scanner.advance_if(any_of!('=')) {
                                self.next_token_to_return =
                                    Some(Token::Punctuator(Punctuator::PercentEqual));
                            } else if self.scanner.advance_if(any_of!('>')) {
                                // digraph
                                self.next_token_to_return =
                                    Some(Token::Punctuator(Punctuator::RightCurlyBracket));
                            } else {
                                self.next_token_to_return =
                                    Some(Token::Punctuator(Punctuator::Percent));
                            }
                            Punctuator::Hash
                        }
                    } else {
                        Punctuator::Hash
                    }
                } else {
                    Punctuator::Percent
                }
            }
            '^' => {
                if self.scanner.advance_if(any_of!('=')) {
                    Punctuator::CaretEqual
                } else {
                    Punctuator::Caret
                }
            }
            ':' => {
                if self.scanner.advance_if(any_of!('>')) {
                    // digraph
                    Punctuator::RightSquareBracket
                } else {
                    Punctuator::Colon
                }
            }
            '=' => {
                if self.scanner.advance_if(any_of!('=')) {
                    Punctuator::EqualEqual
                } else {
                    Punctuator::Equal
                }
            }
            '#' => {
                if self.scanner.advance_if(any_of!('#')) {
                    Punctuator::HashHash
                } else {
                    Punctuator::Hash
                }
            }
            '?' => Punctuator::Question,
            ';' => Punctuator::Semicolon,
            ',' => Punctuator::Comma,
            // Objective-C
            '@' => Punctuator::At,
            c => {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::UnexpectedChar(c),
                    format!("unexpected character {}", c),
                    self.position.clone(),
                ))
            }
        };
        Ok(Token::Punctuator(punc))
    }

    // Should only be called when we're sure there's a next token (or an error)
    fn next_token(&mut self) -> Result<Token, ParseError> {
        let token = if self.scanner.peeking_does_match(char::is_ascii_digit) {
            self.read_number_literal()?
        } else if self.scanner.peeking_does_match(any_of!('\'')) {
            self.read_char_literal(None)?
        } else if self.scanner.peeking_does_match(any_of!('"')) {
            self.read_string_literal(None)?
        } else if let Some(c) = self.scanner.next_if(any_of!('a'..='z', 'A'..='Z', '_')) {
            let mut identifier = String::new();
            identifier.push(c);
            self.scanner.push_while(
                &mut identifier,
                any_of!('a'..='z', 'A'..='Z', '0'..='9', '_'),
            );

            let mut objc_keyword: Option<ObjCKeyword> = None;
            if let Some(Token::Punctuator(Punctuator::At)) = self.previous_token {
                if let Some(&keyword) = OBJC_KEYWORDS.get(identifier.as_str()) {
                    objc_keyword = Some(keyword);
                }
            }

            if let Some(keyword) = objc_keyword {
                Token::ObjCKeyword(keyword)
            } else if let Some(&keyword) = KEYWORDS.get(identifier.as_str()) {
                Token::Keyword(keyword)
            } else {
                match self.scanner.peek() {
                    Some('\'') => {
                        if let Some(prefix) = Self::char_literal_prefix(identifier.as_str()) {
                            self.read_char_literal(Some(prefix))?
                        } else {
                            Token::Identifier(identifier)
                        }
                    }
                    Some('\"') => {
                        if let Some(prefix) = Self::string_literal_prefix(identifier.as_str()) {
                            self.read_string_literal(Some(prefix))?
                        } else {
                            Token::Identifier(identifier)
                        }
                    }
                    _ => Token::Identifier(identifier),
                }
            }
        } else {
            self.read_punctuator()?
        };
        Ok(token)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EndKind {
    Line,
    File,
}

impl<'a> FailableIterator for BasicTokenIter<'a> {
    type Item = PositionedToken;
    type Error = ParseError;

    fn next(&mut self) -> Result<Option<PositionedToken>, ParseError> {
        if let Some(token) = self.next_token_to_return.take() {
            return Ok(Some(PositionedToken(token, self.position.clone())));
        }

        let mut preprocessing_directive: Option<Vec<Token>> = None;

        let token = loop {
            self.scanner.advance_while(any_of!(' ', '\t'));

            let mut end: Option<EndKind> = None;
            if self.scanner.peek().is_none() {
                end = Some(EndKind::File);
            } else if self.scanner.advance_if(any_of!('\n')) {
                end = Some(EndKind::Line);
            } else if self.scanner.advance_if(any_of!('\r')) {
                self.scanner.advance_if(any_of!('\n'));
                end = Some(EndKind::Line);
            };
            if let Some(end) = end {
                if let Some(ref mut directive) = preprocessing_directive {
                    self.handle_preprocessing_directive(directive.as_ref())?;
                    preprocessing_directive = None;
                } else {
                    self.position.line += 1;
                }
                if end == EndKind::File {
                    return Ok(None);
                }
                self.is_start_of_line = true;
                continue;
            }

            let token = self.next_token()?;
            if token == Token::Punctuator(Punctuator::Hash) {
                if !self.is_start_of_line {
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidPreprocDirective,
                        "a processing directive should be at the start of a line".to_string(),
                        self.position.clone(),
                    ));
                } else {
                    preprocessing_directive = Some(Vec::new());
                    self.is_start_of_line = false;
                    continue;
                }
            }
            self.is_start_of_line = false;

            if let Some(ref mut directive) = preprocessing_directive {
                directive.push(token);
            } else {
                break PositionedToken(token, self.position.clone());
            }
        };
        let PositionedToken(ref valid_token, _) = token;
        self.previous_token = Some(valid_token.clone());
        Ok(Some(token))
    }
}

pub struct TokenIter<'a> {
    iter: FailablePeekable<BasicTokenIter<'a>>,
}

impl<'a> TokenIter<'a> {
    pub fn from(code: &'a str) -> TokenIter<'a> {
        TokenIter {
            iter: BasicTokenIter::from(code).peekable(),
        }
    }
}

impl<'a> FailableIterator for TokenIter<'a> {
    type Item = PositionedToken;
    type Error = ParseError;

    fn next(&mut self) -> Result<Option<PositionedToken>, ParseError> {
        Ok(match self.iter.next()? {
            // merge adjacent string literals
            Some(PositionedToken(Token::Literal(Literal::Str(mut literal, mut prefix)), pos)) => {
                loop {
                    match self.iter.peek()? {
                        Some(PositionedToken(Token::Literal(Literal::Str(_, _)), _)) => {
                            if let Some(PositionedToken(
                                Token::Literal(Literal::Str(following_literal, following_prefix)),
                                following_position,
                            )) = self.iter.next()?
                            {
                                if prefix != following_prefix {
                                    if prefix.is_some() {
                                        if following_prefix.is_some() {
                                            return Err(ParseError::new_with_position(
                                                ParseErrorKind::InvalidConstruct,
                                                "adjacent string literals must use the same prefix"
                                                    .to_string(),
                                                following_position,
                                            ));
                                        }
                                    } else {
                                        prefix = following_prefix;
                                    }
                                }
                                literal.push_str(following_literal.as_ref());
                            } else {
                                unreachable!();
                            }
                        }
                        _ => {
                            break Some(PositionedToken(
                                Token::Literal(Literal::Str(literal, prefix)),
                                pos,
                            ))
                        }
                    }
                }
            }
            next => next,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_one_valid_token(code: &str) -> Token {
        let mut iter = TokenIter::from(code);
        let token = iter.next().unwrap().unwrap();
        if let Some(_) = iter.next().unwrap() {
            panic!("Multiple tokens found in {:}", code);
        }
        match token {
            PositionedToken(token, _) => token,
        }
    }

    fn parse_valid_tokens(code: &str) -> Vec<Token> {
        let mut iter = TokenIter::from(code);
        let mut tokens = Vec::new();
        loop {
            match iter.next() {
                Ok(Some(PositionedToken(token, _))) => tokens.push(token),
                Ok(None) => break,
                Err(err) => panic!("Error parsing \"{:}\": {:?}", code, err),
            }
        }
        tokens
    }

    #[test]
    fn test_string_literal() {
        assert_eq!(
            parse_one_valid_token(r#""abcd\"""#),
            Token::Literal(Literal::Str("abcd\"".to_string(), None)),
        );
        assert_eq!(
            parse_one_valid_token(r#""a\tbc\nd""#),
            Token::Literal(Literal::Str("a\tbc\nd".to_string(), None)),
        );
        assert_eq!(
            parse_one_valid_token(r#""\x61\0\0123""#),
            Token::Literal(Literal::Str("a\0\n3".to_string(), None)),
        );
        assert_eq!(
            parse_one_valid_token(r#""abcd" "efgh" "ijkl""#),
            Token::Literal(Literal::Str("abcdefghijkl".to_string(), None)),
        );
        // adjacent string literal must all have the same prefix
        // (if there's no prefix it becomes the prefix or the adjacent string)
        assert_eq!(
            parse_one_valid_token(r#"u"abcd" u"efgh""#),
            Token::Literal(Literal::Str(
                "abcdefgh".to_string(),
                Some(StringPrefix::Char16)
            )),
        );
        assert_eq!(
            parse_one_valid_token(r#""abcd" U"efgh""#),
            Token::Literal(Literal::Str(
                "abcdefgh".to_string(),
                Some(StringPrefix::Char32)
            )),
        );
        assert_eq!(
            parse_one_valid_token(r#"L"abcd" "efgh""#),
            Token::Literal(Literal::Str(
                "abcdefgh".to_string(),
                Some(StringPrefix::WChar)
            )),
        );
    }

    #[test]
    fn test_number_literal() {
        assert_eq!(
            parse_one_valid_token("123"),
            Token::Literal(Literal::Integer("123".to_string(), IntegerRepr::Dec, None)),
        );
        assert_eq!(
            parse_one_valid_token("0x123"),
            Token::Literal(Literal::Integer("123".to_string(), IntegerRepr::Hex, None)),
        );
        assert_eq!(
            parse_one_valid_token("0.123"),
            Token::Literal(Literal::Float("0.123".to_string(), FloatRepr::Dec, None)),
        );
        assert_eq!(
            parse_one_valid_token("0.123f"),
            Token::Literal(Literal::Float(
                "0.123".to_string(),
                FloatRepr::Dec,
                Some(FloatSuffix::Float),
            )),
        );
        assert_eq!(
            parse_one_valid_token("0x0.123p12L"),
            Token::Literal(Literal::Float(
                "0.123p12".to_string(),
                FloatRepr::Hex,
                Some(FloatSuffix::LongDouble),
            )),
        );
    }

    #[test]
    fn test_punctuators() {
        assert_eq!(
            parse_one_valid_token("%:%:"),
            Token::Punctuator(Punctuator::HashHash),
        );
        assert_eq!(
            parse_valid_tokens(".."),
            vec!(
                Token::Punctuator(Punctuator::Period),
                Token::Punctuator(Punctuator::Period)
            ),
        );
        assert_eq!(
            parse_one_valid_token("..."),
            Token::Punctuator(Punctuator::Ellipsis),
        );
        assert_eq!(
            parse_one_valid_token("%:%:"),
            Token::Punctuator(Punctuator::HashHash),
        );
    }
}
