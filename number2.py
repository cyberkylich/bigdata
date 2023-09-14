def check(names, new_nick):
    nick = str(input())
    if nick in names:
        names[nick] += 1
        nick += str(names[nick])
        new_nick.append(nick)
    else:
        names[nick] = 0
        new_nick.append("OK")
    return (new_nick)


def main():
    names = {}
    new_nick = []
    n = input()
    for i in range(int(n)):
        check(names, new_nick)

    for name in new_nick:
        print(name)

    return

if __name__ == "__main__":
        main()