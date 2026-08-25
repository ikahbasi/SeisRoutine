
def cashe_all_available_pretrained_seisbench_models(models):
    for model in models:
        for pretrained in model.list_pretrained():
            print(pretrained)
            try:
                model.from_pretrained(pretrained)
            except Exception as error:
                print(error)