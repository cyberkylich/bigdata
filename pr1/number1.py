morze = {'a': '.-', 'b': '-…', 'c': '-.-.', 'd': '-..',
         'e': '.', 'f': '..-.', 'g': '--.', 'h': '….',
         'i': '..', 'j': '.---', 'k': '-.-', 'l': '.-..',
         'm': '--', 'n': '-.', 'o': '---', 'p': '.--.',
         'q': '--.-', 'r': '.-.', 's': '…', 't': '-',
         'u': '..-', 'v': '…-', 'w': '.--', 'x': '-..-',
         'y': '-.--', 'z': '--..'}

inp = input("Введите текст: ")
inp = inp.lower()
translate_text = ""
for lett in inp:
    if lett != " ":
        translate_text += morze[lett]
    else:
        translate_text += "\n"
print(translate_text)
