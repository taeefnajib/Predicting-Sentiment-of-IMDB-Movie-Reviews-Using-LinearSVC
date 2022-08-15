import pickle
#Loading model
model = pickle.load(open('model/model.sav', 'rb'))
#Question asking loop
while True:
    user_input = str(input("Enter Movie Review: "))
    if user_input=="q":
        break
    output = model.predict([user_input])
    print("Sentiment:",str.capitalize(output[0]),"\n")
    