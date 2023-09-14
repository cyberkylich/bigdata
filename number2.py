def check(names):
    k = 0
    nick = input()
    new_nick = ""
    if nick in names:
        k += 1
        # nick.append(k)
        new_nick = nick + k
        names[new_nick] += k
        print(new_nick)
    else:
        k = 0
        names[nick] = k
        print("OK")
    print(names.items())
    return


def main():
    names = {}
    n = input()
    for i in range(int(n)):
        check(names)
    return

    if __name__ == "__main__":
        main()