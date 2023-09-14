def create_files(files):
    up = str(input())
    up = up.split()
    files[up[0]] = up[1:]
    return(files)

def read_files(files, request):
    action = str(input())
    command = action.split()[0]
    file_name = action.split()[1]
    match command:
        case "read":
            command = "r"
        case "write":
            command = "w"
        case "execute":
            command = "x"
        case _:
            print("wrong command!")
            return
    if command in files[file_name]:
        request.append("OK")
    else:
        request.append("Access denied")
    return(request)



def main():
    files = {}
    request = []
    n = input()
    for i in range(int(n)):
        create_files(files)
    m = input()
    for k in range(int(m)):
        read_files(files,request)
    for record in request:
        print(record)
    return 0


if __name__ == "__main__":
    main()