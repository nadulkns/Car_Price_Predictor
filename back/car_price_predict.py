import json
import pickle
import numpy as np
import sys

__brands = None
__data_columns = None
__model = None
__body_type = None


def load_saved_artifacts():
    print('loading saved artifacts...start')
    global __brands
    global __data_columns
    global __model
    global __body_type

    with open("./model/columns.json", 'r', encoding="utf-8") as f:
        __data_columns = json.load(f)['data_columns']
        __brands = __data_columns[2:19]
        __body_type = __data_columns[21:]

    if __model is None:
        with open("./model/car_prices_model.pickle", 'rb') as f:
            __model = pickle.load(f)
    print('loading saved artifacts...done')


def predict_price(make, fueltype, bodytype, enginesize, horsepower):
    loc_index_make = __data_columns.index(make.lower())
    loc_index_fueltype = __data_columns.index(fueltype.lower())
    loc_index_bodytype = __data_columns.index(bodytype.lower())

    x = np.zeros(len(__data_columns))
    x[0] = enginesize
    x[1] = horsepower
    if loc_index_make >= 0:
        x[loc_index_make] = 1
    if loc_index_fueltype >= 0:
        x[loc_index_fueltype] = 1
    if loc_index_bodytype >= 0:
        x[loc_index_bodytype] = 1

    return np.round(__model.predict([x])[0]*42, 2)


def view_car_brands():
    count = 1
    for brand in __brands:
        print(f"\t{count}.{brand}")
        count = count+1

    main_menu()


def view_car_categories():
    count = 1
    for cat in __body_type:
        print(f"\t{count}.{cat}")
        count = count+1

    main_menu()


def predict_car_price():
    make = input("\nEnter Car Brand :")
    btype = input("Enter Body Type :")
    fuel = input("Enter Fuel Type (gas/diesel) :")
    eng = int(input("Enter Engine Size :"))
    horsep = int(input("Enter Horse Power :"))

    predict = predict_price(make, fuel, btype, eng, horsep)

    print(f"\nEstimated Price is (Rs.){predict} ")

    main_menu()


def main_menu():
    print('\t *** CAR PRICE PREDICTOR ***\n')
    print("Press 1 : View Car Brands ")
    print("Press 2 : View Car Categories ")
    print("Press 3 : View Car Price")
    print("Press 4 : Exit")

    ch = input("\nEnter Your Choice : ")

    match ch:
        case "1":
            view_car_brands()
        case "2":
            view_car_categories()
        case "3":
            predict_car_price()
        case "4":
            sys.exit("End.")


if __name__ == "__main__":
    load_saved_artifacts()
    main_menu()
    # print(predict_price('audi', 'diesel', 'sedan', 200, 175))
