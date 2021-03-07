# coding: utf-8


import unicodedata
import re
from collections import OrderedDict
import json

def strQ2B(string):
    C_pun = u'，！？【】（）《》“”‘’；：．▏‖'
    E_pun = u',!?[]()<>""\'\';:.||'
    table= {ord(f):ord(t) for f,t in zip(C_pun, E_pun)}
    string = string.translate(table)

    # 转换说明：
    # 全角字符unicode编码从65281~65374 （十六进制 0xFF01 ~ 0xFF5E）
    # 半角字符unicode编码从33~126 （十六进制 0x21~ 0x7E）
    # 空格比较特殊，全角为 12288（0x3000），半角为 32（0x20）
    # 除空格外，全角/半角按unicode编码排序在顺序上是对应的（半角 + 0x7e= 全角）,所以可以直接通过用+-法来处理非空格数据，对空格单独处理。
    rstring = ""
    for uchar in string:
        # 返回赋予Unicode字符uchar的字符串型通用分类。
        inside_code = ord(uchar)

        if inside_code == 0 or inside_code == 0xfffd:
            continue
        cat = unicodedata.category(uchar)
        if cat == "Mn" or cat=='Cf' or cat=="So":
            continue
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


compiles = re.compile("\([0-9A-Za-z]\)")
special_token_start_index = 9312  # ①
special_token_start_word = "[unused%s]"
other_special_start_word = "[unused%s]"

chinese = re.compile('[\u4e00-\u9fa5,。:;]+')
alphabet_comiles = re.compile("[A-Za-z]+")
dot_number_compile = re.compile("[1-9]\.")
bracket_compile = re.compile("\([a-zA-Z0-9]\)")
chinese_space =re.compile(r"([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])")# 中文之间的空格
space_compile = re.compile(r'\s+')

def judge_choice_chinses(choices):
    """判断选项中，是否都没有中文"""
    had_chinses = False
    for c in choices:
        if chinese.search(c):
            had_chinses = True
        elif len("".join(number_compile.findall(c))) == len(c):
            had_chinses = True
    return had_chinses

def judge_choice_alphabet(choices, question, context):
    """检查选项中，是否含有字母，如果有，再次检测是否是拼英"""
    is_alpha = False
    is_from_question = False
    all_token_from_context = []
    all_token_from_question = []
    if all([True if alphabet_comiles.search(c) else False for c in choices]) and all(
            [True if c.find("(") == -1 else False for c in choices]):
        """在每个选项中都找到字母"""
        all_alphabet = sum([alphabet_comiles.findall(c) for c in choices], [])
        all_alphabet = set(sum([list(w) for w in all_alphabet], []))
        all_specal_token = all_alphabet
        all_alphabet_index_question = [(question.find(w), w, len(w)) if question.find(w) > -1 else (-1, w, len(w)) for w
                                       in all_alphabet]
        all_alphabet_index_question = sorted(all_alphabet_index_question, key=lambda x: x[0])
        filters = list(filter(lambda x: x[0] != -1, all_alphabet_index_question))
        for i, j in zip(filters[:-1], filters[1:]):
            if j[0] - i[0] < 2:
                is_alpha = True
        all_alphabet_index_context = [(context.find(w), w, len(w)) if context.find(w) > -1 else (-1, w, len(w)) for w in
                                      all_alphabet]
        all_alphabet_index_context = sorted(all_alphabet_index_context, key=lambda x: x[0])
        filters = list(filter(lambda x: x[0] != -1, all_alphabet_index_context))
        for i, j in zip(filters[:-1], filters[1:]):
            if j[0] - i[0] < 2:  # 两个字母在文本很相近，说明其可能是拼英
                is_alpha = True
        if not is_alpha:
            all_token_from_context = all_alphabet_index_context
            all_token_from_question = all_alphabet_index_question

    if is_alpha or (len(all_token_from_context) == 0 and len(all_token_from_question) == 0):
        if any([True if 9312 <= ord(w) < 9362 else False for c in choices for w in c]):
            """每个选项中③④⑤"""
            all_specal_token = set([a for c in choices for a in c if 9312 <= ord(a) < 9362])
            all_special_from_question = [(question.find(w), w, len(w)) if question.find(w) > -1 else (-1, w, len(w)) for
                                         w in all_specal_token]
            all_special_fom_context = [(context.find(w), w, len(w)) if context.find(w) > -1 else (-1, w, len(w)) for w
                                       in all_specal_token]
            all_token_from_question = all_special_from_question
            all_token_from_context = all_special_fom_context
        elif any([True if dot_number_compile.search(c) else False for c in choices]):
            number_compile = re.compile("[1-9]\.?")
            all_specal_token = sum([number_compile.findall(c) for c in choices], [])
            all_number_from_question = [(question.find(w), w, len(w)) if question.find(w) > -1 else (-1, w, len(w)) for
                                        w in all_specal_token]
            all_number_fom_context = [(context.find(w), w, len(w)) if context.find(w) > -1 else (-1, w, len(w)) for w in
                                      all_specal_token]
            all_token_from_question = all_number_from_question
            all_token_from_context = all_number_fom_context
        elif any([True if bracket_compile.search(c) else False for c in choices]):
            all_specal_token = sum([bracket_compile.findall(c) for c in choices], [])
            all_bracket_from_question = [(question.find(w), w, len(w)) if question.find(w) > -1 else (-1, w, len(w)) for
                                         w in all_specal_token]
            all_bracket_fom_context = [(context.find(w), w, len(w)) if context.find(w) > -1 else (-1, w, len(w)) for w
                                       in all_specal_token]
            all_token_from_question = all_bracket_from_question
            all_token_from_context = all_bracket_fom_context
        else:
            raise ("")
    all_token_from_question = list(filter(lambda x: x[0] != -1, all_token_from_question))
    all_token_from_context = list(filter(lambda x: x[0] != -1, all_token_from_context))
    if len(all_token_from_question) == len(all_specal_token):
        all_token_index = all_token_from_question
        is_from_question = True
    elif len(all_token_from_context) == len(all_specal_token):
        all_token_index = all_token_from_context
    else:
        print(all_token_from_context)
        print(all_token_from_question)
        raise ("ValueError")

    return is_alpha, all_token_index, is_from_question


choice_space = re.compile("^[ABCDabcd]\s?(\.|、|。)?")

alpha_compile = re.compile("[a-zA-Z0-9]")
alphabet = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13,
            'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25,
            'z': 26}
number_compile = re.compile('[0-9]+')


def sub_bracket_number(w):
    """将选项中(1)(2) 或者abcd这些编号替换成[unused5？]从51开始的编号"""
    if len(w) != 1:
        w = w.replace(")", "").replace("(", "").replace(".", "")
    w = w.lower()
    if number_compile.search(w):
        num = number_compile.findall(w)[0]
        other_special_word = other_special_start_word % str(50 + int(num))
        return other_special_word
    elif w.lower() in 'abcdefghijklmnopqrstuvwxyz':
        num = alphabet[w]
        other_special_word = other_special_start_word % str(50 + num)
        return other_special_word
    elif special_token_start_index <= ord(w) <= 9362:
        """①④⑥ 这些替换成[unused1？]，从11开始的特殊字符"""
        special_token_index = ord(w) - special_token_start_index + 1
        special_token_word = special_token_start_word % str(10 + special_token_index)
        return special_token_word
    else:
        print(w)
        raise ValueError




def format_processing(lines,is_training=True):

    new_data = []
    will_drop_q_id = ["012602", "274101"]

    for d in lines:
        new_d = OrderedDict()
        text = d['Content']
        text = strQ2B(text)
        ID = d['ID']
        all_token_from_contexts = []
        Questions = []
        for q in d['Questions']:
            new_q = OrderedDict()
            question = q['Question']
            question = strQ2B(question)
            choices = q['Choices']
            if is_training:
                answer_id = q['Answer']
            Q_id = q['Q_id']
            for index, c in enumerate(choices):
                choices[index] = re.sub("\s+", "", choice_space.sub('', strQ2B(c)).strip())
            if not judge_choice_chinses(choices) and q['Q_id'] not in will_drop_q_id:
                new_choices = []
                is_alpha, all_token_from_question, is_from_question = judge_choice_alphabet(choices, question, text)
                if len(all_token_from_question):
                    if is_from_question:
                        # print(all_token_from_question)
                        all_token_from_question = sorted(all_token_from_question, key=lambda x: x[0], reverse=True)
                        all_alphabet = [w for _, w, _ in all_token_from_question]
                        for i, w, length in all_token_from_question:
                            su_word = sub_bracket_number(w)
                            question = question[:int(i)] + su_word + question[int(i) + int(length):]
                    else:
                        all_token_from_context = sorted(all_token_from_question, key=lambda x: x[0], reverse=True)
                        all_alphabet = [w for _, w, _ in all_token_from_context]
                        all_token_from_contexts.extend(all_token_from_context)
                        # print(all_token_from_context)

                    for c in choices:
                        ab = {}
                        for a in all_alphabet:
                            if c.find(a) > -1:
                                c_index = c.find(a)
                                ab[a] = c_index
                        for w, i in sorted(ab.items(), key=lambda x: x[1], reverse=True):
                            c = c[:i] + sub_bracket_number(w) + c[i + len(w):]

                        new_choices.append(space_compile.sub("", chinese_space.sub(r"\1[SPACE]\2",c.lower())))
                else:
                    print(d['ID'])

                    print(choices)
                choices = new_choices

            new_q['Questions'] = space_compile.sub("", chinese_space.sub(r"\1[SPACE]\2",question.lower()))
            new_q['Choices'] = choices
            if is_training:
                new_q['Answer'] = answer_id
            new_q['Q_id'] = Q_id
            Questions.append(new_q)

        if len(all_token_from_contexts):
            all_token_from_contexts = sorted(set(all_token_from_contexts), key=lambda x: x[0], reverse=True)

            for i, w, length in all_token_from_contexts:
                su_word = sub_bracket_number(w)
                if w != text[i:i + length]:
                    # print(d['ID'],text[i:i+length],w)
                    # print(all_token_from_contexts)
                    break
                text = text[:int(i)] + su_word + text[int(i) + int(length):]

        new_d["ID"] = ID
        new_d['Content'] = space_compile.sub("", chinese_space.sub(r"\1[SPACE]\2",re.sub(r"[\n]+",'',text.lower())))
        new_d['Questions'] = Questions
        new_data.append(new_d)
    return new_data

if __name__ == '__main__':
    with open("validation_v1.json", 'r', encoding="utf-8") as f:
        data = json.load(f)
    new_data = format_processing(data,False)

    with open("validation_v2.json", 'w', encoding="utf-8") as f:
        f.write(json.dumps(new_data, ensure_ascii=False))

    with open("train_v1.json", 'r', encoding="utf-8") as f:
        data = json.load(f)
    new_data = format_processing(data)

    with open("train_v2.json", 'w', encoding="utf-8") as f:
        f.write(json.dumps(new_data, ensure_ascii=False))
