import resources

if __name__ == "__main__":
        df = resources.Functions.ImportData()
        while True:
            resources.Options.printMenu()
            option = int(input("Please enter a positive number: "))
            resources.Options.chooseFromMenu(option)
