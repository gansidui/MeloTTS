import re
import unittest
import string

def escape_regex(text):
    # 转义正则表达式中的特殊字符
    special_chars = r"\.+*?()|[]{}^$"
    escaped = []
    for char in text:
        if char in special_chars:
            escaped.append('\\')
        escaped.append(char)
    return ''.join(escaped)

def contains_in_string_array(arr, value):
    # 检查数组中是否包含一个字符串
    return value in arr

def split_text_with_separators(text, separators):
    # 对每个分隔符进行转义，以确保它们在正则表达式中被正确处理
    escaped_separators = [escape_regex(sep) for sep in separators]
    
    # 构建正则表达式以包含所有转义后的分隔符，并确保分隔符在分割后保留
    regex_pattern = f"({'|'.join(escaped_separators)})"
    reg = re.compile(regex_pattern)
    
    # 查找所有分隔符的位置
    matches = list(reg.finditer(text))
    ret = []
    start = 0
    
    # 根据找到的分隔符位置分割文本
    for match in matches:
        end = match.end()
        segment = text[start:end]
        if not contains_in_string_array(separators, segment):
            ret.append(segment)
        start = end

    # 检查并添加最后一部分（如果存在）
    if start < len(text):
        last_segment = text[start:]
        if not contains_in_string_array(separators, last_segment):
            ret.append(last_segment)

    return ret


def is_punctuations(s):
    return all(char in string.punctuation for char in s)

def split_long_text(text, separators):
    segment_array = split_text_with_separators(text, separators)
    ret = []
    for segment in segment_array:
        segment = segment.strip()
        if segment and not is_punctuations(segment):
            ret.append(segment)
    return ret


class TestSplitTextWithSeparators(unittest.TestCase):
    def test_split_text_with_separators(self):
        ss = split_text_with_separators("hello, world", [", "])
        self.assertEqual(ss, ["hello, ", "world"])

        ss = split_text_with_separators("hello.world", ["."])
        self.assertEqual(ss, ["hello.", "world"])

        ss = split_text_with_separators("helloworld", [", "])
        self.assertEqual(ss, ["helloworld"])

        ss = split_text_with_separators("hello..world", ["."])
        self.assertEqual(ss, ["hello.", "world"])

        ss = split_text_with_separators("hello..world.", ["."])
        self.assertEqual(ss, ["hello.", "world."])

        ss = split_text_with_separators("hello.. world..", ["."])
        self.assertEqual(ss, ["hello.", " world."])

        ss = split_text_with_separators("hello.. world..", [". "])
        self.assertEqual(ss, ["hello.. ", "world.."])

        ss = split_text_with_separators("hello..,,world..,,", [".", ","])
        self.assertEqual(ss, ["hello.", "world."])

        ss = split_text_with_separators(", hello..world.", ["."])
        self.assertEqual(ss, [", hello.", "world."])

        ss = split_text_with_separators(", hello..world.", [".", ","])
        self.assertEqual(ss, [" hello.", "world."])

        ss = split_text_with_separators("hello, . world.", [", ", ". "])
        self.assertEqual(ss, ["hello, ", "world."])

        ss = split_text_with_separators("你吃饭了吗？？我爱吃饭，也爱吃肉。真好吃！你呢？", ["，", "。", "？", "！"])
        self.assertEqual(ss, ["你吃饭了吗？", "我爱吃饭，", "也爱吃肉。", "真好吃！", "你呢？"])

        ss = split_text_with_separators("how are you, i'm fine, thank you.", [", "])
        self.assertEqual(ss, ["how are you, ", "i'm fine, ", "thank you."])

        ss = split_text_with_separators("吃饭、洗澡、睡觉", ["、"])
        self.assertEqual(ss, ["吃饭、", "洗澡、", "睡觉"])

        ss = split_text_with_separators("hello، world", ["، "])
        self.assertEqual(ss, ["hello، ", "world"])

        ss = split_text_with_separators("hello، ، world", ["، "])
        self.assertEqual(ss, ["hello، ", "world"])

        ss = split_text_with_separators("hello। world", ["। "])
        self.assertEqual(ss, ["hello। ", "world"])

        ss = split_text_with_separators("hello। । world", ["। "])
        self.assertEqual(ss, ["hello। ", "world"])


class TestSplitLongText(unittest.TestCase):
    def test_split_long_text(self):
        ss = split_long_text("我爱吃饭，也吃蔬菜。", ["，"])
        self.assertEqual(ss, ["我爱吃饭，", "也吃蔬菜。"])

        ss = split_long_text("hello.world", ["."])
        self.assertEqual(ss, ["hello.", "world"])

        ss = split_long_text("helloworld", ["."])
        self.assertEqual(ss, ["helloworld"])

        ss = split_long_text("how, are, you.", [", "])
        self.assertEqual(ss, ["how,", "are,", "you."])

        ss = split_long_text("hello, .  world.", [", ", ". "])
        self.assertEqual(ss, ["hello,", "world."])

        ss = split_long_text("10,000", [", "])
        self.assertEqual(ss, ["10,000"])


if __name__ == '__main__':
    unittest.main()
