from .classifier import AbstractClassifier


class Errors(AbstractClassifier):
    def __init__(self):
        self.pattern = ["(get(\w{0,4})|got|throw(\w{0,3})|threw|show(\w{0,3})|giv(\w{0,3})|gave|hav(\w{0,3})|had|see(\w{0,3})|display(\w{0,3})|catch(\w{0,3})|(un)?caught|rece?ive)( [^\s]+){0,5} (error|(\w{0,35})exception)",
                        "(error|exceptions?)( [^\s]+){0,5} (get(\w{0,3})|got|throw(\w{0,3})|show(\w{0,3})|giv(\w{0,3})|gave|hav(\w{0,3})|had|display(\w{0,3})|catch(\w{0,3})|caught)",
                        "(log\s?cat|stack\s?trace|log|print\s?traceback)( [^\s]+){0,5} (error|exception)",
                        "(errors?|exceptions?)( [^\s]+){0,5} (logcat|stacktrace|log|message)",
                        "((re)?solve|fix)( [^\s]+){0,5} (error|(\w{0,35})exception)",
                        "(TypeError|AttributeError|KeyError|NameError|ValueError|SyntaxError|IndentationError|ImportError|ModuleNotFoundError|IndexError|AssertionError|RuntimeError|NotImplementedError)",
                        "(NameError: name '[^\s]+' is not defined)",
                        "(IndentationError: unexpected indent|IndentationError: expected an indented block)",
                        "(SyntaxError: invalid syntax)",
                        "(TypeError: [^\s]+ object is not subscriptable)",
                        "(AttributeError: '([^\s]+)' object has no attribute '([^\s]+)')",
                        "(KeyError: '[^\s]+')",
                        "(ValueError: [^\s]+)",
                        "(IndexError: list index out of range)",
                        "(AssertionError)",
                        "(FileNotFoundError)",
                        "(ModuleNotFoundError)",
                        "(ImportError: cannot import name '[^\s]+')",
                        "(OSError: \[Errno [0-9]+\] [^\s]+: '[^\s]+'|PermissionError: \[Errno [0-9]+\] [^\s]+: '[^\s]+')",
                        "(TypeError: unsupported operand type\(s\) for [^\s]+: '[^\s]+' and '[^\s]+')",
                        "(TypeError: [^\s]+() takes [0-9]+ positional argument[s]* but [0-9]+ were given)",
                        "(TypeError: [^\s]+() missing [0-9]+ required positional argument[s]*: '[^\s]+')",
                        "(TypeError: [^\s]+() got an unexpected keyword argument '[^\s])",
                        "(TypeError: can only concatenate [^\s]+ \(not '[^\s]+'\) to [^\s]+)",
                        "(TypeError: [^\s]+ object is not callable)",
                        "(TypeError: 'str' object is not callable)"
                        "(TypeError: 'int' object is not subscriptable)",
                        "(TypeError: 'NoneType' object is not subscriptable)",
                        "(TypeError: cannot unpack non-iterable [^\s]+ object)",
                        "(TypeError: unhashable type: '[^\s]+')",
                        "(Name|Attribute|TypeError|Index|Key)?Error(:)? [^\n]+",
                        "[^\\w]assert(ion)? +[^\n]+",
                        "File \".+\", line [0-9]+, [^\n]+",
                        "builtins.(Value)?Error(:)? [^\n]+",
                        "builtins.(Import)?Warning(:)? [^\n]+",
                        "Traceback (most recent call last):.+",
                        "(Module|Syntax|Indentation|Tab)?Error(:)? [^\n]+",
                        "[^\\w]exit\\([0-9]+\\)",
                        "RecursionError(:)? [^\n]+",
                        "IOError: +[^\n]+",
                        "OSError: +[^\n]+",
                        "PermissionError: +[^\n]+",
                        "RecursionError: +maximum recursion depth exceeded",
                        "MemoryError: +[^\n]+",
                        "KeyboardInterrupt",
                        "SystemExit"]

        self.antipattern = [
            "(not get(\w{0,4})|no) (error|exception)",
            "(don't|dont|does not|doesnt|doesn't|no)( [^\s]+){0,3} (crash|throw)",
        ]
        return

    @property
    def name(self):
        return self.__class__.__name__.upper()

    def classify(self, title, question):
        # dlog(id, ":  ", rr)
        count_yes = 0
        count_no = 0

        support = []
        for pat in self.pattern:
            found, m = self.countMatches(pat, title + " " + question)
            count_yes += found
            if found > 0:
                support.append([found, pat])
                # print(m)

        for pat in self.antipattern:
            found, m = self.countMatches(pat, title + " " + question)
            count_no += found
            if found > 0:
                support.append([(-1) * found, pat])

        points = count_yes - count_no
        if points > 0:

            return 1
        else:

            return 0
