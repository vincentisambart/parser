// TODO:
// - tests
// - for punctuators, use a match
// - move punctuators code to a different method
// - make file path in Position non-optional
// - add Objective-C keywords (look at clang's TokenKinds.def)
// - handle \r \n \t... (look at clang's LiteralSupport.cpp)
// - maybe use Rc for strings in tokens
// - only one pass

use lazy_static::lazy_static;
use std::collections::HashMap;
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

    fn scan_two<F1, F2>(&mut self, mut matcher1: F1, mut matcher2: F2) -> Option<(char, char)>
    where
        F1: FnMut(&char) -> bool,
        F2: FnMut(&char) -> bool,
    {
        let mut tmp = self.chars.clone();
        let c1 = match tmp.next() {
            Some(c) => c,
            None => return None,
        };
        if !matcher1(&c1) {
            return None;
        }
        let c2 = match tmp.next() {
            Some(c) => c,
            None => return None,
        };
        if !matcher2(&c2) {
            return None;
        }
        self.chars = tmp;
        Some((c1, c2))
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

#[derive(Debug, Clone, Copy)]
enum IntegerRepr {
    Dec,
    Hex,
    Oct,
}

#[derive(Debug, Clone, Copy)]
enum FloatRepr {
    Dec,
    Hex,
}

#[derive(Debug, Clone, Copy)]
enum IntegerSuffix {
    Unsigned,
    Long,
    LongLong,
    UnsignedLong,
    UnsignedLongLong,
}

#[derive(Debug, Clone, Copy)]
enum FloatSuffix {
    Float,
    LongDouble,
}

#[derive(Debug, Clone, Copy)]
enum CharPrefix {
    WChar,
    Char16,
    Char32,
}

#[derive(Debug, Clone, Copy)]
enum StringPrefix {
    Utf8,
    WChar,
    Char16,
    Char32,
}

#[derive(Debug, Clone, Copy)]
enum Punctuator {
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
enum LexError {
    UnexpectedEOF,
    UnexpectedChar(char),
    InvalidPreprocessorDirective(Position),
}

#[derive(Debug, Clone)]
enum RawToken {
    IntegerLiteral(String, IntegerRepr, Option<IntegerSuffix>),
    FloatLiteral(String, FloatRepr, Option<FloatSuffix>),
    CharLiteral(char, Option<CharPrefix>),
    StringLiteral(String, Option<StringPrefix>),
    Identifier(String),
    Keyword(Keyword),
    Punctuator(Punctuator),
    NewLine,
}

struct RawTokenIter<'a> {
    scanner: CharsScanner<'a>,
}

impl<'a> RawTokenIter<'a> {
    fn from(text: &'a str) -> RawTokenIter<'a> {
        let scanner = CharsScanner::from(text);
        RawTokenIter { scanner }
    }

    fn read_number_literal(&mut self) -> Result<RawToken, LexError> {
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
                    return Ok(RawToken::FloatLiteral(token, FloatRepr::Hex, suffix));
                } else {
                    let suffix = self.read_integer_suffix();
                    return Ok(RawToken::IntegerLiteral(token, IntegerRepr::Hex, suffix));
                }
            } else if let Some(digit) = self.scanner.scan_one(CharPattern::is_ascii_octdigit) {
                // octal
                token.push(digit);
                self.scanner
                    .scan_append(CharPattern::is_ascii_octdigit, &mut token);
                return Ok(RawToken::IntegerLiteral(
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
            self.scanner
                .scan_append(char::is_ascii_hexdigit, &mut token);
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
            Ok(RawToken::FloatLiteral(token, FloatRepr::Dec, suffix))
        } else {
            let suffix = self.read_integer_suffix();
            Ok(RawToken::IntegerLiteral(token, IntegerRepr::Dec, suffix))
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

    fn read_char_escape(&mut self) -> Result<char, LexError> {
        if self.scanner.scan_one(any_of!('x')).is_some() {
            panic!("TODO");
        } else if self
            .scanner
            .scan_one(CharPattern::is_ascii_octdigit)
            .is_some()
        {
            panic!("TODO");
        } else if let Some(c) = self.scanner.next() {
            Ok(c)
        } else {
            Err(LexError::UnexpectedEOF)
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

    fn read_char_literal(&mut self, prefix: Option<CharPrefix>) -> Result<RawToken, LexError> {
        self.scanner.scan_one(any_of!('\'')).unwrap();
        let t = if self.scanner.scan_one(any_of!('\\')).is_some() {
            let c = self.read_char_escape()?;
            RawToken::CharLiteral(c, prefix)
        } else if let Some(c) = self.scanner.next() {
            RawToken::CharLiteral(c, prefix)
        } else {
            return Err(LexError::UnexpectedEOF);
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

    fn read_string_literal(&mut self, prefix: Option<StringPrefix>) -> Result<RawToken, LexError> {
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
                return Err(LexError::UnexpectedEOF);
            };
        }

        Ok(RawToken::StringLiteral(string, prefix))
    }
}

impl<'a> Iterator for RawTokenIter<'a> {
    type Item = Result<RawToken, LexError>;

    fn next(&mut self) -> Option<Result<RawToken, LexError>> {
        // skip whitespace (new lines have special treatment)
        self.scanner.skip_while(any_of!(' ', '\t'));

        if self.scanner.peek().is_none() {
            None
        } else if self.scanner.check_one(char::is_ascii_digit).is_some() {
            Some(self.read_number_literal())
        } else if self.scanner.scan_one(any_of!('\n')).is_some() {
            Some(Ok(RawToken::NewLine))
        } else if self.scanner.scan_one(any_of!('\r')).is_some() {
            self.scanner.scan_one(any_of!('\n'));
            Some(Ok(RawToken::NewLine))
        } else if self.scanner.check_one(any_of!('\'')).is_some() {
            Some(self.read_char_literal(None))
        } else if self.scanner.check_one(any_of!('"')).is_some() {
            Some(self.read_string_literal(None))
        } else if let Some(c) = self.scanner.scan_one(any_of!('a'..='z', 'A'..='Z', '_')) {
            let mut identifier = String::new();
            identifier.push(c);
            self.scanner.scan_append(
                any_of!('a'..='z', 'A'..='Z', '0'..='9', '_'),
                &mut identifier,
            );

            if let Some(&keyword) = KEYWORDS.get(identifier.as_str()) {
                Some(Ok(RawToken::Keyword(keyword)))
            } else {
                match self.scanner.peek() {
                    Some('\'') => {
                        if let Some(prefix) = RawTokenIter::char_literal_prefix(identifier.as_str())
                        {
                            Some(self.read_char_literal(Some(prefix)))
                        } else {
                            Some(Ok(RawToken::Identifier(identifier)))
                        }
                    }
                    Some('\"') => {
                        if let Some(prefix) =
                            RawTokenIter::string_literal_prefix(identifier.as_str())
                        {
                            Some(self.read_string_literal(Some(prefix)))
                        } else {
                            Some(Ok(RawToken::Identifier(identifier)))
                        }
                    }
                    _ => Some(Ok(RawToken::Identifier(identifier))),
                }
            }
        } else if self.scanner.scan_one(any_of!('[')).is_some() {
            Some(Ok(RawToken::Punctuator(Punctuator::LeftSquareBracket)))
        } else if self.scanner.scan_one(any_of!(']')).is_some() {
            Some(Ok(RawToken::Punctuator(Punctuator::RightSquareBracket)))
        } else if self.scanner.scan_one(any_of!('(')).is_some() {
            Some(Ok(RawToken::Punctuator(Punctuator::LeftParenthesis)))
        } else if self.scanner.scan_one(any_of!(')')).is_some() {
            Some(Ok(RawToken::Punctuator(Punctuator::RightParenthesis)))
        } else if self.scanner.scan_one(any_of!('{')).is_some() {
            Some(Ok(RawToken::Punctuator(Punctuator::LeftCurlyBracket)))
        } else if self.scanner.scan_one(any_of!('}')).is_some() {
            Some(Ok(RawToken::Punctuator(Punctuator::RightCurlyBracket)))
        } else if self.scanner.scan_one(any_of!('-')).is_some() {
            if self.scanner.scan_one(any_of!('-')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::MinusMinus)))
            } else if self.scanner.scan_one(any_of!('>')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::Arrow)))
            } else if self.scanner.scan_one(any_of!('=')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::MinusEqual)))
            } else {
                Some(Ok(RawToken::Punctuator(Punctuator::Minus)))
            }
        } else if self.scanner.scan_one(any_of!('+')).is_some() {
            if self.scanner.scan_one(any_of!('+')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::PlusPlus)))
            } else if self.scanner.scan_one(any_of!('=')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::PlusEqual)))
            } else {
                Some(Ok(RawToken::Punctuator(Punctuator::Plus)))
            }
        } else if self.scanner.scan_one(any_of!('&')).is_some() {
            if self.scanner.scan_one(any_of!('&')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::AmpersandAmpersand)))
            } else if self.scanner.scan_one(any_of!('=')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::AmpersandEqual)))
            } else {
                Some(Ok(RawToken::Punctuator(Punctuator::Ampersand)))
            }
        } else if self.scanner.scan_one(any_of!('|')).is_some() {
            if self.scanner.scan_one(any_of!('|')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::PipePipe)))
            } else if self.scanner.scan_one(any_of!('=')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::PipeEqual)))
            } else {
                Some(Ok(RawToken::Punctuator(Punctuator::Pipe)))
            }
        } else if self.scanner.scan_one(any_of!('<')).is_some() {
            if self.scanner.scan_one(any_of!('<')).is_some() {
                if self.scanner.scan_one(any_of!('=')).is_some() {
                    Some(Ok(RawToken::Punctuator(Punctuator::LessLessEqual)))
                } else {
                    Some(Ok(RawToken::Punctuator(Punctuator::LessLess)))
                }
            } else if self.scanner.scan_one(any_of!('=')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::LessEqual)))
            } else if self.scanner.scan_one(any_of!(':')).is_some() {
                // digraph
                Some(Ok(RawToken::Punctuator(Punctuator::LeftSquareBracket)))
            } else if self.scanner.scan_one(any_of!('%')).is_some() {
                // digraph
                Some(Ok(RawToken::Punctuator(Punctuator::LeftCurlyBracket)))
            } else {
                Some(Ok(RawToken::Punctuator(Punctuator::Less)))
            }
        } else if self.scanner.scan_one(any_of!('>')).is_some() {
            if self.scanner.scan_one(any_of!('>')).is_some() {
                if self.scanner.scan_one(any_of!('=')).is_some() {
                    Some(Ok(RawToken::Punctuator(Punctuator::GreaterGreaterEqual)))
                } else {
                    Some(Ok(RawToken::Punctuator(Punctuator::GreaterGreater)))
                }
            } else if self.scanner.scan_one(any_of!('=')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::GreaterEqual)))
            } else {
                Some(Ok(RawToken::Punctuator(Punctuator::Greater)))
            }
        } else if self.scanner.scan_one(any_of!('.')).is_some() {
            if self.scanner.scan_two(any_of!('.'), any_of!('.')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::Ellipsis)))
            } else {
                Some(Ok(RawToken::Punctuator(Punctuator::Period)))
            }
        } else if self.scanner.scan_one(any_of!('*')).is_some() {
            if self.scanner.scan_one(any_of!('=')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::StarEqual)))
            } else {
                Some(Ok(RawToken::Punctuator(Punctuator::Star)))
            }
        } else if self.scanner.scan_one(any_of!('~')).is_some() {
            Some(Ok(RawToken::Punctuator(Punctuator::Tilde)))
        } else if self.scanner.scan_one(any_of!('!')).is_some() {
            if self.scanner.scan_one(any_of!('=')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::ExclamationEqual)))
            } else {
                Some(Ok(RawToken::Punctuator(Punctuator::Exclamation)))
            }
        } else if self.scanner.scan_one(any_of!('/')).is_some() {
            if self.scanner.scan_one(any_of!('=')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::SlashEqual)))
            } else {
                Some(Ok(RawToken::Punctuator(Punctuator::Slash)))
            }
        } else if self.scanner.scan_one(any_of!('%')).is_some() {
            if self.scanner.scan_one(any_of!('=')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::PercentEqual)))
            } else if self.scanner.scan_one(any_of!('>')).is_some() {
                // digraph
                Some(Ok(RawToken::Punctuator(Punctuator::RightCurlyBracket)))
            } else if self.scanner.scan_one(any_of!(':')).is_some() {
                // digraph
                if self.scanner.scan_two(any_of!(':'), any_of!('>')).is_some() {
                    Some(Ok(RawToken::Punctuator(Punctuator::HashHash)))
                } else {
                    Some(Ok(RawToken::Punctuator(Punctuator::Hash)))
                }
            } else {
                Some(Ok(RawToken::Punctuator(Punctuator::Percent)))
            }
        } else if self.scanner.scan_one(any_of!('^')).is_some() {
            if self.scanner.scan_one(any_of!('=')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::CaretEqual)))
            } else {
                Some(Ok(RawToken::Punctuator(Punctuator::Caret)))
            }
        } else if self.scanner.scan_one(any_of!(':')).is_some() {
            if self.scanner.scan_one(any_of!('>')).is_some() {
                // digraph
                Some(Ok(RawToken::Punctuator(Punctuator::RightSquareBracket)))
            } else {
                Some(Ok(RawToken::Punctuator(Punctuator::Colon)))
            }
        } else if self.scanner.scan_one(any_of!('=')).is_some() {
            if self.scanner.scan_one(any_of!('=')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::EqualEqual)))
            } else {
                Some(Ok(RawToken::Punctuator(Punctuator::Equal)))
            }
        } else if self.scanner.scan_one(any_of!('#')).is_some() {
            if self.scanner.scan_one(any_of!('#')).is_some() {
                Some(Ok(RawToken::Punctuator(Punctuator::HashHash)))
            } else {
                Some(Ok(RawToken::Punctuator(Punctuator::Hash)))
            }
        } else if self.scanner.scan_one(any_of!('?')).is_some() {
            Some(Ok(RawToken::Punctuator(Punctuator::Question)))
        } else if self.scanner.scan_one(any_of!(';')).is_some() {
            Some(Ok(RawToken::Punctuator(Punctuator::Semicolon)))
        } else if self.scanner.scan_one(any_of!(',')).is_some() {
            Some(Ok(RawToken::Punctuator(Punctuator::Comma)))
        } else if self.scanner.scan_one(any_of!('@')).is_some() {
            // Objective-C
            Some(Ok(RawToken::Punctuator(Punctuator::At)))
        } else {
            Some(Err(LexError::UnexpectedChar(self.scanner.peek().unwrap())))
        }
    }
}

#[derive(Debug, Clone)]
struct Position {
    file_path: Option<String>,
    line: u32,
}

#[derive(Debug, Clone, Copy)]
enum Keyword {
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
}

#[derive(Debug, Clone)]
enum Token {
    IntegerLiteral(String, IntegerRepr, Option<IntegerSuffix>, Position),
    FloatLiteral(String, FloatRepr, Option<FloatSuffix>, Position),
    CharLiteral(char, Option<CharPrefix>, Position),
    StringLiteral(String, Option<StringPrefix>, Position),
    Identifier(String, Position),
    Punctuator(Punctuator, Position),
    Keyword(Keyword, Position),
}

struct TokenIter<'a> {
    raw_iter: RawTokenIter<'a>,
    position: Position,
}

impl<'a> TokenIter<'a> {
    fn from(text: &'a str) -> TokenIter<'a> {
        let raw_iter = RawTokenIter::from(text);
        let position = Position {
            file_path: None,
            line: 1,
        };
        TokenIter { raw_iter, position }
    }

    fn read_preprocessor_directive(&mut self) -> Result<(), LexError> {
        let mut token = match self.raw_iter.next() {
            None => {
                return Err(LexError::InvalidPreprocessorDirective(
                    self.position.clone(),
                ))
            }
            Some(Err(error)) => return Err(error),
            Some(Ok(token)) => token,
        };
        if let RawToken::Identifier(name) = token {
            if name == "line" {
                token = match self.raw_iter.next() {
                    None => {
                        return Err(LexError::InvalidPreprocessorDirective(
                            self.position.clone(),
                        ))
                    }
                    Some(Err(error)) => return Err(error),
                    Some(Ok(token)) => token,
                };
            } else {
                return Err(LexError::InvalidPreprocessorDirective(
                    self.position.clone(),
                ));
            }
        }

        let line;
        if let RawToken::IntegerLiteral(text, IntegerRepr::Dec, None) = token {
            line = u32::from_str_radix(text.as_ref(), 10).unwrap();
        } else {
            return Err(LexError::InvalidPreprocessorDirective(
                self.position.clone(),
            ));
        }

        token = match self.raw_iter.next() {
            None => {
                return Err(LexError::InvalidPreprocessorDirective(
                    self.position.clone(),
                ))
            }
            Some(Err(error)) => return Err(error),
            Some(Ok(token)) => token,
        };

        let file_path;
        if let RawToken::StringLiteral(text, None) = token {
            file_path = text;
        } else {
            return Err(LexError::InvalidPreprocessorDirective(
                self.position.clone(),
            ));
        }
        self.position.line = line;
        self.position.file_path = Some(file_path);

        loop {
            match self.raw_iter.next() {
                None | Some(Ok(RawToken::NewLine)) => return Ok(()),
                Some(Err(error)) => return Err(error),
                Some(Ok(RawToken::IntegerLiteral(_, _, _))) => (),
                Some(Ok(_)) => {
                    return Err(LexError::InvalidPreprocessorDirective(
                        self.position.clone(),
                    ))
                }
            }
        }
    }
}

impl<'a> Iterator for TokenIter<'a> {
    type Item = Result<Token, LexError>;

    fn next(&mut self) -> Option<Result<Token, LexError>> {
        loop {
            let token = match self.raw_iter.next() {
                None => return None,
                Some(Ok(token)) => token,
                Some(Err(error)) => return Some(Err(error)),
            };
            match token {
                RawToken::NewLine => self.position.line += 1,
                RawToken::IntegerLiteral(text, repr, suffix) => {
                    return Some(Ok(Token::IntegerLiteral(
                        text,
                        repr,
                        suffix,
                        self.position.clone(),
                    )))
                }
                RawToken::FloatLiteral(text, repr, suffix) => {
                    return Some(Ok(Token::FloatLiteral(
                        text,
                        repr,
                        suffix,
                        self.position.clone(),
                    )));
                }
                RawToken::CharLiteral(c, prefix) => {
                    return Some(Ok(Token::CharLiteral(c, prefix, self.position.clone())));
                }
                RawToken::StringLiteral(text, prefix) => {
                    return Some(Ok(Token::StringLiteral(
                        text,
                        prefix,
                        self.position.clone(),
                    )));
                }
                RawToken::Keyword(keyword) => {
                    return Some(Ok(Token::Keyword(keyword, self.position.clone())));
                }
                RawToken::Identifier(identifier) => {
                    return Some(Ok(Token::Identifier(identifier, self.position.clone())));
                }
                RawToken::Punctuator(Punctuator::Hash) => {
                    if let Err(error) = self.read_preprocessor_directive() {
                        return Some(Err(error));
                    }
                }
                RawToken::Punctuator(punctuator) => {
                    return Some(Ok(Token::Punctuator(punctuator, self.position.clone())));
                }
            }
        }
    }
}

fn main() -> Result<(), LexError> {
    let iter = TokenIter::from(
        r#"# 1 "test.c"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 341 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "test.c" 2
# 1 "./test.h" 1
int foo(void);
# 2 "test.c" 2
int foo(void) {
  return 10;
}
"#,
    );
    for token in iter {
        println!("token: {:?}", token?);
    }
    Ok(())
}
