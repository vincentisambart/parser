// TODO:
// - tests
// - for punctuators, use a match
// - move punctuators code to a different method
// - maybe use Rc for strings in tokens
// - maybe support universal character names
// - maybe support other non-ascii identifiers (check what clang does)
// - handle more preprocessing directives (#error, unknown #pragma should be an error...)
// - maybe use io::Read instead of std::Chars
// - do not go to end when error but make sure to skip the error character to not risking ending in an infinite loop

use lazy_static::lazy_static;
use std::char;
use std::collections::HashMap;
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

struct CharsScanner<'a> {
    chars: Chars<'a>,
}

impl<'a> CharsScanner<'a> {
    fn from(text: &'a str) -> CharsScanner<'a> {
        CharsScanner {
            chars: text.chars(),
        }
    }

    fn peek(&mut self) -> Option<char> {
        let old = self.chars.clone();
        let opt = self.chars.next();
        self.chars = old;
        opt
    }

    fn skip_while<F>(&mut self, mut matcher: F)
    where
        F: FnMut(&char) -> bool,
    {
        loop {
            let old = self.chars.clone();
            let c = match self.chars.next() {
                Some(c) => c,
                None => break,
            };

            if !matcher(&c) {
                self.chars = old;
                break;
            }
        }
    }

    fn next(&mut self) -> Option<char> {
        self.chars.next()
    }

    fn scan_one<F>(&mut self, mut matcher: F) -> Option<char>
    where
        F: FnMut(&char) -> bool,
    {
        let old = self.chars.clone();
        let c = match self.chars.next() {
            Some(c) => c,
            None => return None,
        };

        if matcher(&c) {
            Some(c)
        } else {
            self.chars = old;
            None
        }
    }

    fn check_one<F>(&mut self, mut matcher: F) -> Option<char>
    where
        F: FnMut(&char) -> bool,
    {
        let mut tmp = self.chars.clone();
        let c = match tmp.next() {
            Some(c) => c,
            None => return None,
        };
        if matcher(&c) {
            Some(c)
        } else {
            None
        }
    }

    fn scan_append<F>(&mut self, mut matcher: F, string: &mut String)
    where
        F: FnMut(&char) -> bool,
    {
        loop {
            let old = self.chars.clone();

            match self.chars.next() {
                Some(c) => {
                    if matcher(&c) {
                        string.push(c);
                    } else {
                        self.chars = old;
                        break;
                    }
                }
                None => break,
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerRepr {
    Dec,
    Hex,
    Oct,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
    Auto,
    Break,
    Case,
    Char,
    Const,
    Continue,
    Default,
    Do,
    Double,
    Else,
    Enum,
    Extern,
    Float,
    For,
    Goto,
    If,
    Inline,
    Int,
    Long,
    Register,
    Restrict,
    Return,
    Short,
    Signed,
    Sizeof,
    Static,
    Struct,
    Switch,
    Typedef,
    Union,
    Unsigned,
    Void,
    Volatile,
    While,
    Alignas,
    Alignof,
    Atomic,
    Bool,
    Complex,
    Generic,
    Imaginary,
    Noreturn,
    StaticAssert,
    ThreadLocal,
}

// Objective-C keywords, always after a @.
// The list comes from clang's include/clang/Basic/TokenKinds.def.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjCKeyword {
    Class,
    CompatibilityAlias,
    Defs,
    Encode,
    End,
    Implementation,
    Interface,
    Private,
    Protected,
    Protocol,
    Public,
    Selector,
    Throw,
    Try,
    Catch,
    Finally,
    Synchronized,
    Autoreleasepool,
    Property,
    Package,
    Required,
    Optional,
    Synthesize,
    Dynamic,
    Import,
    Available,
}

lazy_static! {
    static ref KEYWORDS: HashMap<&'static str, Keyword> = {
        let mut keywords = HashMap::new();

        keywords.insert("auto", Keyword::Auto);
        keywords.insert("break", Keyword::Break);
        keywords.insert("case", Keyword::Case);
        keywords.insert("char", Keyword::Char);
        keywords.insert("const", Keyword::Const);
        keywords.insert("continue", Keyword::Continue);
        keywords.insert("default", Keyword::Default);
        keywords.insert("do", Keyword::Do);
        keywords.insert("double", Keyword::Double);
        keywords.insert("else", Keyword::Else);
        keywords.insert("enum", Keyword::Enum);
        keywords.insert("extern", Keyword::Extern);
        keywords.insert("float", Keyword::Float);
        keywords.insert("for", Keyword::For);
        keywords.insert("goto", Keyword::Goto);
        keywords.insert("if", Keyword::If);
        keywords.insert("inline", Keyword::Inline);
        keywords.insert("int", Keyword::Int);
        keywords.insert("long", Keyword::Long);
        keywords.insert("register", Keyword::Register);
        keywords.insert("restrict", Keyword::Restrict);
        keywords.insert("return", Keyword::Return);
        keywords.insert("short", Keyword::Short);
        keywords.insert("signed", Keyword::Signed);
        keywords.insert("sizeof", Keyword::Sizeof);
        keywords.insert("static", Keyword::Static);
        keywords.insert("struct", Keyword::Struct);
        keywords.insert("switch", Keyword::Switch);
        keywords.insert("typedef", Keyword::Typedef);
        keywords.insert("union", Keyword::Union);
        keywords.insert("unsigned", Keyword::Unsigned);
        keywords.insert("void", Keyword::Void);
        keywords.insert("volatile", Keyword::Volatile);
        keywords.insert("while", Keyword::While);
        keywords.insert("_Alignas", Keyword::Alignas);
        keywords.insert("_Alignof", Keyword::Alignof);
        keywords.insert("_Atomic", Keyword::Atomic);
        keywords.insert("_Bool", Keyword::Bool);
        keywords.insert("_Complex", Keyword::Complex);
        keywords.insert("_Generic", Keyword::Generic);
        keywords.insert("_Imaginary", Keyword::Imaginary);
        keywords.insert("_Noreturn", Keyword::Noreturn);
        keywords.insert("_Static_assert", Keyword::StaticAssert);
        keywords.insert("_Thread_local", Keyword::ThreadLocal);

        keywords
    };
    static ref OBJC_KEYWORDS: HashMap<&'static str, ObjCKeyword> = {
        let mut keywords = HashMap::new();

        keywords.insert("class", ObjCKeyword::Class);
        keywords.insert("compatibility_alias", ObjCKeyword::CompatibilityAlias);
        keywords.insert("defs", ObjCKeyword::Defs);
        keywords.insert("encode", ObjCKeyword::Encode);
        keywords.insert("end", ObjCKeyword::End);
        keywords.insert("implementation", ObjCKeyword::Implementation);
        keywords.insert("interface", ObjCKeyword::Interface);
        keywords.insert("private", ObjCKeyword::Private);
        keywords.insert("protected", ObjCKeyword::Protected);
        keywords.insert("protocol", ObjCKeyword::Protocol);
        keywords.insert("public", ObjCKeyword::Public);
        keywords.insert("selector", ObjCKeyword::Selector);
        keywords.insert("throw", ObjCKeyword::Throw);
        keywords.insert("try", ObjCKeyword::Try);
        keywords.insert("catch", ObjCKeyword::Catch);
        keywords.insert("finally", ObjCKeyword::Finally);
        keywords.insert("synchronized", ObjCKeyword::Synchronized);
        keywords.insert("autoreleasepool", ObjCKeyword::Autoreleasepool);
        keywords.insert("property", ObjCKeyword::Property);
        keywords.insert("package", ObjCKeyword::Package);
        keywords.insert("required", ObjCKeyword::Required);
        keywords.insert("optional", ObjCKeyword::Optional);
        keywords.insert("synthesize", ObjCKeyword::Synthesize);
        keywords.insert("dynamic", ObjCKeyword::Dynamic);
        keywords.insert("import", ObjCKeyword::Import);
        keywords.insert("available", ObjCKeyword::Available);

        keywords
    };
}

#[derive(Debug, Clone)]
pub enum LexError {
    UnexpectedEOF(Position),
    UnexpectedChar(char, Position),
    InvalidPreprocessingDirective(Position),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    IntegerLiteral(String, IntegerRepr, Option<IntegerSuffix>),
    FloatLiteral(String, FloatRepr, Option<FloatSuffix>),
    CharLiteral(char, Option<CharPrefix>),
    StringLiteral(String, Option<StringPrefix>),
    Identifier(String),
    Keyword(Keyword),
    ObjCKeyword(ObjCKeyword),
    Punctuator(Punctuator),
}

#[derive(Debug, Clone)]
pub struct PositionedToken {
    token: Token,
    position: Position,
}

impl PositionedToken {
    fn new(token: Token, position: Position) -> PositionedToken {
        PositionedToken { token, position }
    }

    pub fn token(&self) -> Token {
        self.token.clone()
    }

    pub fn position(&self) -> Position {
        self.position.clone()
    }
}

pub struct TokenIter<'a> {
    scanner: CharsScanner<'a>,
    position: Position,
    is_start_of_line: bool,
    previous_token: Option<Token>,
    next_token_to_return: Option<Token>,
}

impl<'a> TokenIter<'a> {
    pub fn from(code: &'a str) -> TokenIter<'a> {
        let scanner = CharsScanner::from(code);
        let position = Position {
            file_path: Rc::new("<unknown>".to_string()),
            line: 1,
        };
        TokenIter {
            scanner,
            position,
            is_start_of_line: true,
            previous_token: None,
            next_token_to_return: None,
        }
    }

    fn read_number_literal(&mut self) -> Result<Token, LexError> {
        let mut token = String::new();

        if self.scanner.scan_one(any_of!('0')).is_some() {
            if self.scanner.scan_one(any_of!('x', 'X')).is_some() {
                // hexadecimal
                self.scanner
                    .scan_append(char::is_ascii_hexdigit, &mut token);
                assert!(!token.is_empty());
                let mut is_float = false;
                if self.scanner.scan_one(any_of!('.')).is_some() {
                    is_float = true;
                    token.push('.');
                    self.scanner
                        .scan_append(char::is_ascii_hexdigit, &mut token);
                }
                if let Some(p) = self.scanner.scan_one(any_of!('p', 'P')) {
                    is_float = true;
                    token.push(p);
                    if let Some(sign) = self.scanner.scan_one(any_of!('-', '+')) {
                        token.push(sign);
                    }
                    self.scanner.scan_append(char::is_ascii_digit, &mut token);
                }
                if is_float {
                    let suffix = self.read_float_suffix();
                    return Ok(Token::FloatLiteral(token, FloatRepr::Hex, suffix));
                } else {
                    let suffix = self.read_integer_suffix();
                    return Ok(Token::IntegerLiteral(token, IntegerRepr::Hex, suffix));
                }
            } else if let Some(digit) = self.scanner.scan_one(CharPattern::is_ascii_octdigit) {
                // octal
                token.push(digit);
                self.scanner
                    .scan_append(CharPattern::is_ascii_octdigit, &mut token);
                return Ok(Token::IntegerLiteral(
                    token,
                    IntegerRepr::Oct,
                    self.read_integer_suffix(),
                ));
            }

            token.push('0');
        }

        self.scanner.scan_append(char::is_ascii_digit, &mut token);
        let mut is_float = false;
        if self.scanner.scan_one(any_of!('.')).is_some() {
            is_float = true;
            token.push('.');
            self.scanner.scan_append(char::is_ascii_digit, &mut token);
            if let Some(e) = self.scanner.scan_one(any_of!('e', 'E')) {
                token.push(e);
                if let Some(sign) = self.scanner.scan_one(any_of!('-', '+')) {
                    token.push(sign);
                }
                self.scanner.scan_append(char::is_ascii_digit, &mut token);
            }
        }
        if let Some(e) = self.scanner.scan_one(any_of!('e', 'E')) {
            is_float = true;
            token.push(e);
            if let Some(sign) = self.scanner.scan_one(any_of!('-', '+')) {
                token.push(sign);
            }
            self.scanner.scan_append(char::is_ascii_digit, &mut token);
        }
        if is_float {
            let suffix = self.read_float_suffix();
            Ok(Token::FloatLiteral(token, FloatRepr::Dec, suffix))
        } else {
            let suffix = self.read_integer_suffix();
            Ok(Token::IntegerLiteral(token, IntegerRepr::Dec, suffix))
        }
    }

    fn read_integer_suffix(&mut self) -> Option<IntegerSuffix> {
        if self.scanner.scan_one(any_of!('l', 'L')).is_some() {
            if self.scanner.scan_one(any_of!('l', 'L')).is_some() {
                Some(IntegerSuffix::LongLong)
            } else {
                Some(IntegerSuffix::Long)
            }
        } else if self.scanner.scan_one(any_of!('u', 'U')).is_some() {
            if self.scanner.scan_one(any_of!('l', 'L')).is_some() {
                if self.scanner.scan_one(any_of!('l', 'L')).is_some() {
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
        if self.scanner.scan_one(any_of!('f', 'F')).is_some() {
            Some(FloatSuffix::Float)
        } else if self.scanner.scan_one(any_of!('l', 'L')).is_some() {
            Some(FloatSuffix::LongDouble)
        } else {
            None
        }
    }

    // The list of escapes comes from clang's LiteralSupport.cpp
    fn read_char_escape(&mut self) -> Result<char, LexError> {
        match self.scanner.next() {
            None => Err(LexError::UnexpectedEOF(self.position.clone())),
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
                while let Some(c) = self.scanner.scan_one(char::is_ascii_hexdigit) {
                    let digit = c.to_digit(16).unwrap();
                    x = (x << 4) | digit;
                }
                Ok(char::from_u32(x).unwrap())
            }
            // octal escape
            Some(c) if c.is_ascii_octdigit() => {
                let mut x: u32 = c.to_digit(8).unwrap();
                let mut len = 1;
                while let Some(c) = self.scanner.scan_one(CharPattern::is_ascii_octdigit) {
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

    fn read_char_literal(&mut self, prefix: Option<CharPrefix>) -> Result<Token, LexError> {
        self.scanner.scan_one(any_of!('\'')).unwrap();
        let t = if self.scanner.scan_one(any_of!('\\')).is_some() {
            let c = self.read_char_escape()?;
            Token::CharLiteral(c, prefix)
        } else if let Some(c) = self.scanner.next() {
            Token::CharLiteral(c, prefix)
        } else {
            return Err(LexError::UnexpectedEOF(self.position.clone()));
        };
        self.scanner.scan_one(any_of!('\'')).unwrap();
        Ok(t)
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

    fn read_string_literal(&mut self, prefix: Option<StringPrefix>) -> Result<Token, LexError> {
        self.scanner.scan_one(any_of!('"')).unwrap();
        let mut string = String::new();

        loop {
            if self.scanner.scan_one(any_of!('"')).is_some() {
                break;
            } else if self.scanner.scan_one(any_of!('\\')).is_some() {
                let c = self.read_char_escape()?;
                string.push(c);
            } else if let Some(c) = self.scanner.next() {
                string.push(c);
            } else {
                return Err(LexError::UnexpectedEOF(self.position.clone()));
            };
        }

        Ok(Token::StringLiteral(string, prefix))
    }

    fn handle_preprocessing_directive(&mut self, directive: &[Token]) -> Result<(), LexError> {
        let mut iter = directive.iter();
        match iter.next() {
            None => (), // empty preprocessing directive
            Some(Token::Identifier(identifier)) => match identifier.as_str() {
                // Handle "#line 1 file" as "#1 file"
                "line" => return self.handle_preprocessing_directive(&directive[1..]),
                _ => {
                    return Err(LexError::InvalidPreprocessingDirective(
                        self.position.clone(),
                    ))
                }
            },
            Some(Token::IntegerLiteral(literal, IntegerRepr::Dec, _)) => {
                let line = if let Ok(number) = u32::from_str_radix(literal.as_ref(), 10) {
                    number
                } else {
                    return Err(LexError::InvalidPreprocessingDirective(
                        self.position.clone(),
                    ));
                };

                let file_path;
                if let Some(Token::StringLiteral(text, None)) = iter.next() {
                    file_path = text;
                } else {
                    return Err(LexError::InvalidPreprocessingDirective(
                        self.position.clone(),
                    ));
                }
                self.position.line = line;
                self.position.file_path = Rc::new(file_path.clone());

                loop {
                    match iter.next() {
                        None => break,
                        Some(Token::IntegerLiteral(_, _, _)) => (),
                        Some(_) => {
                            return Err(LexError::InvalidPreprocessingDirective(
                                self.position.clone(),
                            ))
                        }
                    }
                }
            }
            _ => {
                return Err(LexError::InvalidPreprocessingDirective(
                    self.position.clone(),
                ))
            }
        }
        Ok(())
    }

    // Should only be called when we're sure there's a next token (or an error)
    fn next_token(&mut self) -> Result<Token, LexError> {
        let token = if self.scanner.check_one(char::is_ascii_digit).is_some() {
            self.read_number_literal()?
        } else if self.scanner.check_one(any_of!('\'')).is_some() {
            self.read_char_literal(None)?
        } else if self.scanner.check_one(any_of!('"')).is_some() {
            self.read_string_literal(None)?
        } else if let Some(c) = self.scanner.scan_one(any_of!('a'..='z', 'A'..='Z', '_')) {
            let mut identifier = String::new();
            identifier.push(c);
            self.scanner.scan_append(
                any_of!('a'..='z', 'A'..='Z', '0'..='9', '_'),
                &mut identifier,
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
                        if let Some(prefix) = TokenIter::char_literal_prefix(identifier.as_str()) {
                            self.read_char_literal(Some(prefix))?
                        } else {
                            Token::Identifier(identifier)
                        }
                    }
                    Some('\"') => {
                        if let Some(prefix) = TokenIter::string_literal_prefix(identifier.as_str())
                        {
                            self.read_string_literal(Some(prefix))?
                        } else {
                            Token::Identifier(identifier)
                        }
                    }
                    _ => Token::Identifier(identifier),
                }
            }
        } else if self.scanner.scan_one(any_of!('[')).is_some() {
            Token::Punctuator(Punctuator::LeftSquareBracket)
        } else if self.scanner.scan_one(any_of!(']')).is_some() {
            Token::Punctuator(Punctuator::RightSquareBracket)
        } else if self.scanner.scan_one(any_of!('(')).is_some() {
            Token::Punctuator(Punctuator::LeftParenthesis)
        } else if self.scanner.scan_one(any_of!(')')).is_some() {
            Token::Punctuator(Punctuator::RightParenthesis)
        } else if self.scanner.scan_one(any_of!('{')).is_some() {
            Token::Punctuator(Punctuator::LeftCurlyBracket)
        } else if self.scanner.scan_one(any_of!('}')).is_some() {
            Token::Punctuator(Punctuator::RightCurlyBracket)
        } else if self.scanner.scan_one(any_of!('-')).is_some() {
            if self.scanner.scan_one(any_of!('-')).is_some() {
                Token::Punctuator(Punctuator::MinusMinus)
            } else if self.scanner.scan_one(any_of!('>')).is_some() {
                Token::Punctuator(Punctuator::Arrow)
            } else if self.scanner.scan_one(any_of!('=')).is_some() {
                Token::Punctuator(Punctuator::MinusEqual)
            } else {
                Token::Punctuator(Punctuator::Minus)
            }
        } else if self.scanner.scan_one(any_of!('+')).is_some() {
            if self.scanner.scan_one(any_of!('+')).is_some() {
                Token::Punctuator(Punctuator::PlusPlus)
            } else if self.scanner.scan_one(any_of!('=')).is_some() {
                Token::Punctuator(Punctuator::PlusEqual)
            } else {
                Token::Punctuator(Punctuator::Plus)
            }
        } else if self.scanner.scan_one(any_of!('&')).is_some() {
            if self.scanner.scan_one(any_of!('&')).is_some() {
                Token::Punctuator(Punctuator::AmpersandAmpersand)
            } else if self.scanner.scan_one(any_of!('=')).is_some() {
                Token::Punctuator(Punctuator::AmpersandEqual)
            } else {
                Token::Punctuator(Punctuator::Ampersand)
            }
        } else if self.scanner.scan_one(any_of!('|')).is_some() {
            if self.scanner.scan_one(any_of!('|')).is_some() {
                Token::Punctuator(Punctuator::PipePipe)
            } else if self.scanner.scan_one(any_of!('=')).is_some() {
                Token::Punctuator(Punctuator::PipeEqual)
            } else {
                Token::Punctuator(Punctuator::Pipe)
            }
        } else if self.scanner.scan_one(any_of!('<')).is_some() {
            if self.scanner.scan_one(any_of!('<')).is_some() {
                if self.scanner.scan_one(any_of!('=')).is_some() {
                    Token::Punctuator(Punctuator::LessLessEqual)
                } else {
                    Token::Punctuator(Punctuator::LessLess)
                }
            } else if self.scanner.scan_one(any_of!('=')).is_some() {
                Token::Punctuator(Punctuator::LessEqual)
            } else if self.scanner.scan_one(any_of!(':')).is_some() {
                // digraph
                Token::Punctuator(Punctuator::LeftSquareBracket)
            } else if self.scanner.scan_one(any_of!('%')).is_some() {
                // digraph
                Token::Punctuator(Punctuator::LeftCurlyBracket)
            } else {
                Token::Punctuator(Punctuator::Less)
            }
        } else if self.scanner.scan_one(any_of!('>')).is_some() {
            if self.scanner.scan_one(any_of!('>')).is_some() {
                if self.scanner.scan_one(any_of!('=')).is_some() {
                    Token::Punctuator(Punctuator::GreaterGreaterEqual)
                } else {
                    Token::Punctuator(Punctuator::GreaterGreater)
                }
            } else if self.scanner.scan_one(any_of!('=')).is_some() {
                Token::Punctuator(Punctuator::GreaterEqual)
            } else {
                Token::Punctuator(Punctuator::Greater)
            }
        } else if self.scanner.scan_one(any_of!('.')).is_some() {
            if self.scanner.scan_one(any_of!('.')).is_some() {
                if self.scanner.scan_one(any_of!('.')).is_some() {
                    Token::Punctuator(Punctuator::Ellipsis)
                } else {
                    self.next_token_to_return = Some(Token::Punctuator(Punctuator::Period));
                    Token::Punctuator(Punctuator::Period)
                }
            } else {
                Token::Punctuator(Punctuator::Period)
            }
        } else if self.scanner.scan_one(any_of!('*')).is_some() {
            if self.scanner.scan_one(any_of!('=')).is_some() {
                Token::Punctuator(Punctuator::StarEqual)
            } else {
                Token::Punctuator(Punctuator::Star)
            }
        } else if self.scanner.scan_one(any_of!('~')).is_some() {
            Token::Punctuator(Punctuator::Tilde)
        } else if self.scanner.scan_one(any_of!('!')).is_some() {
            if self.scanner.scan_one(any_of!('=')).is_some() {
                Token::Punctuator(Punctuator::ExclamationEqual)
            } else {
                Token::Punctuator(Punctuator::Exclamation)
            }
        } else if self.scanner.scan_one(any_of!('/')).is_some() {
            if self.scanner.scan_one(any_of!('=')).is_some() {
                Token::Punctuator(Punctuator::SlashEqual)
            } else {
                Token::Punctuator(Punctuator::Slash)
            }
        } else if self.scanner.scan_one(any_of!('%')).is_some() {
            if self.scanner.scan_one(any_of!('=')).is_some() {
                Token::Punctuator(Punctuator::PercentEqual)
            } else if self.scanner.scan_one(any_of!('>')).is_some() {
                // digraph
                Token::Punctuator(Punctuator::RightCurlyBracket)
            } else if self.scanner.scan_one(any_of!(':')).is_some() {
                // digraph
                if self.scanner.scan_one(any_of!('%')).is_some() {
                    if self.scanner.scan_one(any_of!(':')).is_some() {
                        Token::Punctuator(Punctuator::HashHash)
                    } else {
                        if self.scanner.scan_one(any_of!('=')).is_some() {
                            self.next_token_to_return =
                                Some(Token::Punctuator(Punctuator::PercentEqual));
                        } else if self.scanner.scan_one(any_of!('>')).is_some() {
                            // digraph
                            self.next_token_to_return =
                                Some(Token::Punctuator(Punctuator::RightCurlyBracket));
                        } else {
                            self.next_token_to_return =
                                Some(Token::Punctuator(Punctuator::Percent));
                        }
                        Token::Punctuator(Punctuator::Hash)
                    }
                } else {
                    Token::Punctuator(Punctuator::Hash)
                }
            } else {
                Token::Punctuator(Punctuator::Percent)
            }
        } else if self.scanner.scan_one(any_of!('^')).is_some() {
            if self.scanner.scan_one(any_of!('=')).is_some() {
                Token::Punctuator(Punctuator::CaretEqual)
            } else {
                Token::Punctuator(Punctuator::Caret)
            }
        } else if self.scanner.scan_one(any_of!(':')).is_some() {
            if self.scanner.scan_one(any_of!('>')).is_some() {
                // digraph
                Token::Punctuator(Punctuator::RightSquareBracket)
            } else {
                Token::Punctuator(Punctuator::Colon)
            }
        } else if self.scanner.scan_one(any_of!('=')).is_some() {
            if self.scanner.scan_one(any_of!('=')).is_some() {
                Token::Punctuator(Punctuator::EqualEqual)
            } else {
                Token::Punctuator(Punctuator::Equal)
            }
        } else if self.scanner.scan_one(any_of!('#')).is_some() {
            if self.scanner.scan_one(any_of!('#')).is_some() {
                Token::Punctuator(Punctuator::HashHash)
            } else {
                Token::Punctuator(Punctuator::Hash)
            }
        } else if self.scanner.scan_one(any_of!('?')).is_some() {
            Token::Punctuator(Punctuator::Question)
        } else if self.scanner.scan_one(any_of!(';')).is_some() {
            Token::Punctuator(Punctuator::Semicolon)
        } else if self.scanner.scan_one(any_of!(',')).is_some() {
            Token::Punctuator(Punctuator::Comma)
        } else if self.scanner.scan_one(any_of!('@')).is_some() {
            // Objective-C
            Token::Punctuator(Punctuator::At)
        } else {
            return Err(LexError::UnexpectedChar(
                self.scanner.peek().unwrap(),
                self.position.clone(),
            ));
        };
        Ok(token)
    }
}

impl<'a> Iterator for TokenIter<'a> {
    type Item = Result<PositionedToken, LexError>;

    fn next(&mut self) -> Option<Result<PositionedToken, LexError>> {
        if let Some(token) = self.next_token_to_return.take() {
            return Some(Ok(PositionedToken::new(token, self.position.clone())));
        }

        let mut preprocessing_directive: Option<Vec<Token>> = None;

        let token = loop {
            self.scanner.skip_while(any_of!(' ', '\t'));

            let mut is_end_of_line = false;
            let mut is_end_of_file = false;
            if self.scanner.peek().is_none() {
                is_end_of_file = true;
            } else if self.scanner.scan_one(any_of!('\n')).is_some() {
                is_end_of_line = true;
            } else if self.scanner.scan_one(any_of!('\r')).is_some() {
                self.scanner.scan_one(any_of!('\n'));
                is_end_of_line = true;
            };
            if is_end_of_line || is_end_of_file {
                if let Some(ref mut directive) = preprocessing_directive {
                    if let Err(err) = self.handle_preprocessing_directive(directive.as_ref()) {
                        break Some(Err(err));
                    };
                    preprocessing_directive = None;
                } else {
                    self.position.line += 1;
                }
                if is_end_of_file {
                    break None;
                }
                self.is_start_of_line = true;
                continue;
            }

            let token = match self.next_token() {
                Ok(token) => token,
                Err(err) => {
                    break Some(Err(err));
                }
            };
            if token == Token::Punctuator(Punctuator::Hash) {
                if !self.is_start_of_line {
                    break Some(Err(LexError::InvalidPreprocessingDirective(
                        self.position.clone(),
                    )));
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
                break Some(Ok(PositionedToken::new(token, self.position.clone())));
            }
        };
        match token {
            Some(Err(_)) => {
                // If an error occurs, go to the end of the file so you don't risk looping endlessly.
                while let Some(_) = self.next() {}
            }
            Some(Ok(ref valid_token)) => {
                self.previous_token = Some(valid_token.token.clone());
            }
            None => (),
        }
        token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_one_valid_token(code: &str) -> Token {
        let mut iter = TokenIter::from(code);
        let token = iter.next().unwrap().unwrap();
        if let Some(_) = iter.next() {
            panic!("Multiple tokens found in {:}", code);
        }
        token.token
    }

    fn parse_valid_tokens(code: &str) -> Vec<Token> {
        let iter = TokenIter::from(code);
        iter.map(|token| token.unwrap().token).collect()
    }

    #[test]
    fn test_string_literal() {
        assert_eq!(
            parse_one_valid_token(r#""abcd\"""#),
            Token::StringLiteral("abcd\"".to_string(), None),
        );
        assert_eq!(
            parse_one_valid_token(r#""a\tbc\nd""#),
            Token::StringLiteral("a\tbc\nd".to_string(), None),
        );
        assert_eq!(
            parse_one_valid_token(r#""\x61\0\0123""#),
            Token::StringLiteral("a\0\n3".to_string(), None),
        );
    }

    #[test]
    fn test_number_literal() {
        assert_eq!(
            parse_one_valid_token("123"),
            Token::IntegerLiteral("123".to_string(), IntegerRepr::Dec, None),
        );
        assert_eq!(
            parse_one_valid_token("0x123"),
            Token::IntegerLiteral("123".to_string(), IntegerRepr::Hex, None),
        );
        assert_eq!(
            parse_one_valid_token("0.123"),
            Token::FloatLiteral("0.123".to_string(), FloatRepr::Dec, None),
        );
        assert_eq!(
            parse_one_valid_token("0.123f"),
            Token::FloatLiteral(
                "0.123".to_string(),
                FloatRepr::Dec,
                Some(FloatSuffix::Float),
            ),
        );
        assert_eq!(
            parse_one_valid_token("0x0.123p12L"),
            Token::FloatLiteral(
                "0.123p12".to_string(),
                FloatRepr::Hex,
                Some(FloatSuffix::LongDouble),
            ),
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
