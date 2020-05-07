using System;
using System.Collections.Generic;
using System.Text;
using Ex03.GarageLogic;

namespace Ex03.ConsoleUI
{
    internal static class UserInterface
    {
        private const string k_MainMenu =
     @"Please choose from the following options and then press ENTER (1-9):
        1. Add a new vehicle to the garage.
        2. Display all vehicle's registration plates in the garage.
        3. Change a vehicle's status.
        4. Inflate a vehicle's wheels to maximum.
        5. Fill up a fuel-powered vehicle.
        6. Charge an electric-powered vehicle.
        7. Display vehicle's full information.
        8. Quit.
        ";

        private static Garage s_Garage = new Garage();
        private static bool isGarageClose = false;

        private enum eMenuOption
        {
            AddVehicle,
            DisplayRegistrationPlates,
            ChangeVehicleStatus,
            InflateVehicleWheels,
            FillUpFuelVehicle,
            ChargeElectricVehicle,
            DisplayVehicleProperties,
            Quit
        }

        public static void Run()
        {
            eMenuOption menuOptionEnum;
            int userMenuSelection;
            string inputAsString;
            while (!isGarageClose)
            {
                try
                {
                    Console.Clear();
                    showMainMenu();
                    inputAsString = Console.ReadLine();
                    userMenuSelection = parseUserSelection(inputAsString, Enum.GetNames(typeof(eMenuOption)).Length);
                    menuOptionEnum = (eMenuOption)Enum.GetValues(typeof(eMenuOption)).GetValue(userMenuSelection - 1);
                    executeMenuSelection(menuOptionEnum);
                }
                catch (FormatException)
                {
                    Console.WriteLine("Error: Invalid input format");
                }
                catch (ValueOutOfRangeException exception)
                {
                    Console.WriteLine(exception.Message);
                }
                catch (ArgumentException exception)
                {
                    Console.WriteLine(exception.Message);
                }
                catch (Exception)
                {
                    Console.WriteLine("Error: exiting program");
                    isGarageClose = true;
                }

                Console.WriteLine("Press any key to continue.");
                Console.ReadLine();
            }
        }

        private static void executeMenuSelection(eMenuOption i_MenuOption)
        {
            switch (i_MenuOption)
            {
                case eMenuOption.AddVehicle:
                    addVehicleToGarage();
                    break;
                case eMenuOption.DisplayRegistrationPlates:
                    displayRegistrationPlates();
                    break;
                case eMenuOption.ChangeVehicleStatus:
                    changeVehicleStatus();
                    break;
                case eMenuOption.InflateVehicleWheels:
                    inflateVehicleWheels();
                    break;
                case eMenuOption.FillUpFuelVehicle:
                    fillFuelInVehicle();
                    break;
                case eMenuOption.ChargeElectricVehicle:
                    chargeElectricVehicle();
                    break;
                case eMenuOption.DisplayVehicleProperties:
                    displayVehicleProperties();
                    break;
                case eMenuOption.Quit:
                    exitGarage();
                    break;
                default:
                    break;
            }
        }

        private static void displayRegistrationPlates()
        {
            string message;
            string userInputAsString;
            List<string> registrationPlate;
            message = string.Format(
                       "Do you want to use filter? :{0}Enter 1 for 'yes' or 2 for 'no'.",
                        Environment.NewLine);
            Console.WriteLine(message);
            userInputAsString = Console.ReadLine();
            while (userInputAsString != "1" && userInputAsString != "2")
            {
                Console.WriteLine(message);
                userInputAsString = Console.ReadLine();
            }

            if (userInputAsString.Equals("1"))
            {
                GarageENums.eVehicleStatus filter = (GarageENums.eVehicleStatus)getUserChoiceFromEnumValues(typeof(GarageENums.eVehicleStatus));
                registrationPlate = s_Garage.GetRegistrationPlate(filter);
                if (registrationPlate.Count == 0)
                {
                    Console.WriteLine(String.Format("There are no vehicles in the garage with status {0}", filter));
                }
            }
            else
            {
                registrationPlate = s_Garage.GetRegistrationPlate();
                if (registrationPlate.Count == 0)
                {
                    Console.WriteLine("The garage is empty");
                }
            }

            if (registrationPlate.Count > 0)
            {
                foreach (string currentPlate in registrationPlate)
                {
                    Console.WriteLine(currentPlate);
                }
            }
        }

        private static void changeVehicleStatus()
        {
            Console.WriteLine("Please enter registration plate number");
            string plateNumber = Console.ReadLine();
            GarageENums.eVehicleStatus newVehicleStatus = (GarageENums.eVehicleStatus)getUserChoiceFromEnumValues(typeof(GarageENums.eVehicleStatus));
            s_Garage.ChangeVehicleStatus(plateNumber, newVehicleStatus);
            Console.WriteLine("Vehicle status changed successfully");
        }

        private static void inflateVehicleWheels()
        {
            Console.WriteLine("Please enter registration plate number");
            string plateNumber = Console.ReadLine();
            s_Garage.InflateVehicleWheelsToMax(plateNumber);
        }

        private static void fillFuelInVehicle()
        {
            float fuelToFill;
            Console.WriteLine("Please enter registration plate number: ");
            string plateNumber = Console.ReadLine();
            if (s_Garage.GetVehicleEnergyType(plateNumber) is ElectricBattery)
            {
                Console.WriteLine(string.Format("An electric vehicle, please choose another vehicle"));
            }
            else
            {
                Energy.eEnergyType energyType = (Energy.eEnergyType)getUserChoiceFromEnumValues(typeof(FuelTank.eFuelType));
                Console.WriteLine("Please enter how many liters of fuel to add: ");
                string fuelAmountAsString = Console.ReadLine();
                if (!float.TryParse(fuelAmountAsString, out fuelToFill))
                {
                    throw new FormatException();
                }

                s_Garage.FuelOrChargeVehicle(plateNumber, energyType, fuelToFill);
                Console.WriteLine("Fuel was added successfully");
            }
        }

        private static void chargeElectricVehicle()
        {
            float chargeAmount;
            Console.WriteLine("Please enter registration plate number: ");
            string plateNumber = Console.ReadLine();
            Console.WriteLine("Please enter how many minutes to charge: ");
            string chargeAmountAsString = Console.ReadLine();
            if (!float.TryParse(chargeAmountAsString, out chargeAmount))
            {
                throw new FormatException();
            }

            s_Garage.FuelOrChargeVehicle(plateNumber, Energy.eEnergyType.Electricity, chargeAmount);
            Console.WriteLine("Charged successfully");
        }

        private static void displayVehicleProperties()
        {
            Console.WriteLine("Please enter registration plate number: ");
            string plateNumber = Console.ReadLine();
            Console.WriteLine(s_Garage.GetVehicleData(plateNumber));
        }

        private static void exitGarage()
        {
            Console.WriteLine("Thank you for using our garage program , have a nice day!");
            isGarageClose = true;
        }

        private static void addVehicleToGarage()
        {
            bool vehicleIsAlreadyInGarage;
            Vehicle vehicleToAdd;
            vehicleToAdd = getVehicleFromUser();
            vehicleIsAlreadyInGarage = s_Garage.AddVehicleToGarage(vehicleToAdd);

            if (vehicleIsAlreadyInGarage)
            {
                Console.WriteLine(string.Format("A vehicle with that registration plate number already exists, choose another option"));
            }
            else
            {
                Console.WriteLine("Vehicle was added succesfully");
            }
        }

        private static Vehicle getVehicleFromUser()
        {
            Vehicle createdVehicle = null;
            List<object> userEnteredProperties = null;
            GarageENums.eVehicleType typeOfVehicleToAdd = (GarageENums.eVehicleType)getUserChoiceFromEnumValues(typeof(GarageENums.eVehicleType));

            switch (typeOfVehicleToAdd)
            {
                case GarageENums.eVehicleType.FuelCar:
                    userEnteredProperties = getPropertiesFromUser(FuelCar.GetRequiredProperties(), 1);
                    break;
                case GarageENums.eVehicleType.ElectricCar:
                    userEnteredProperties = getPropertiesFromUser(ElectricCar.GetRequiredProperties(), 2);
                    break;
                case GarageENums.eVehicleType.FuelMotorcycle:
                    userEnteredProperties = getPropertiesFromUser(FuelMotorcycle.GetRequiredProperties(), 3);
                    break;
                case GarageENums.eVehicleType.ElectricMotorcycle:
                    userEnteredProperties = getPropertiesFromUser(ElectricMotorcycle.GetRequiredProperties(), 4);
                    break;

                case GarageENums.eVehicleType.FuelTruck:
                    userEnteredProperties = getPropertiesFromUser(FuelTruck.GetRequiredProperties(), 5);
                    break;
                default:
                    break;
            }

            createdVehicle = VehicleGenerator.GenerateVehicle(typeOfVehicleToAdd, userEnteredProperties);

            return createdVehicle;
        }

        private static List<object> getPropertiesFromUser(List<VehicleProperties> i_VehicleRequiredProperties, int i_VehicleType)
        {
            List<object> userEnteredProperties = new List<object>();
            string message;
            string userInputAsString;

            foreach (VehicleProperties currentProperty in i_VehicleRequiredProperties)
            {
                if (currentProperty.Type == typeof(bool))
                {
                    message = string.Format(
                                           "Please choose {1}:{0}Enter 1 for 'yes' or 2 for 'no'.",
                                            Environment.NewLine,
                                            currentProperty.Description);
                    Console.WriteLine(message);
                    userInputAsString = Console.ReadLine();
                    while (userInputAsString != "1" && userInputAsString != "2")
                    {
                        Console.WriteLine(message);
                        userInputAsString = Console.ReadLine();
                    }

                    if (userInputAsString.Equals("1"))
                    {
                        userEnteredProperties.Add(true);
                    }
                    else if (userInputAsString.Equals("2"))
                    {
                        userEnteredProperties.Add(false);
                    }
                }
                else if (currentProperty.Type == typeof(int))
                {
                    int value = 0;
                    bool isNumber = false;
                    message = string.Format("Please enter {0} (an integer):", currentProperty.Description);
                    while (!isNumber)
                    {
                        Console.WriteLine(message);
                        userInputAsString = Console.ReadLine();
                        isNumber = int.TryParse(userInputAsString, out value);
                        if (isNumber)
                        {
                            userEnteredProperties.Add(value);
                        }
                    }
                }
                else if (currentProperty.Description == "Energy Level")
                {
                    float energyLevelValue = 0;
                    message = string.Format("Please enter {0} (a number):", currentProperty.Description);
                    Console.WriteLine(message);
                    energyLevelValue = Validation.GetEnergyLevel(i_VehicleType, s_Garage);
                    userEnteredProperties.Add(energyLevelValue);
                }
                else if (currentProperty.Description == "Current Air Pressure")
                {
                    float CurrentPressureValue = 0;
                    message = string.Format("Please enter {0} (a number):", currentProperty.Description);
                    Console.WriteLine(message);
                    CurrentPressureValue = Validation.GetCurrentAirPressure(i_VehicleType, s_Garage);
                    userEnteredProperties.Add(CurrentPressureValue);
                }
                else if (currentProperty.Type == typeof(float))
                {
                    float value = 0;
                    bool isNumber = false;
                    message = string.Format("Please enter {0} (a number):", currentProperty.Description);
                    while (!isNumber)
                    {
                        Console.WriteLine(message);
                        userInputAsString = Console.ReadLine();
                        isNumber = float.TryParse(userInputAsString, out value);
                        if (isNumber)
                        {
                            userEnteredProperties.Add(value);
                        }
                    }
                }
                else if (currentProperty.Type == typeof(string))
                {
                    message = string.Format("Please enter {0}: ", currentProperty.Description);
                    Console.WriteLine(message);
                    userInputAsString = Console.ReadLine();
                    userEnteredProperties.Add(userInputAsString);
                }
                else if (currentProperty.Type.IsEnum)
                {
                    Console.WriteLine(
                                    string.Format(
                                    "Please enter {0}.",
                                    currentProperty.Description));
                    userEnteredProperties.Add(getUserChoiceFromEnumValues(currentProperty.Type));
                }
            }

            return userEnteredProperties;
        }

        private static object getUserChoiceFromEnumValues(Type i_Enum)
        {
            object enumValueToReturn = null;
            if (!i_Enum.IsEnum)
            {
                throw new ArgumentException();
            }

            Console.WriteLine("Please choose one of the following: ");
            int currentValueIndex = 1;

            foreach (object enumValue in Enum.GetValues(i_Enum))
            {
                Console.WriteLine(string.Format("{0} - {1}", currentValueIndex, enumValue));
                currentValueIndex++;
            }

            string textualIndexOfEnumValue = Console.ReadLine();
            int userInputtedIndexOfEnumValue;
            bool isNumber = int.TryParse(textualIndexOfEnumValue, out userInputtedIndexOfEnumValue);

            if (!isNumber)
            {
                throw new FormatException();
            }

            if (userInputtedIndexOfEnumValue < 1 || userInputtedIndexOfEnumValue > currentValueIndex - 1)
            {
                throw new ValueOutOfRangeException(1, currentValueIndex - 1);
            }

            currentValueIndex = 1;
            foreach (object enumValue in Enum.GetValues(i_Enum))
            {
                if (userInputtedIndexOfEnumValue == currentValueIndex)
                {
                    enumValueToReturn = enumValue;
                    break;
                }
                currentValueIndex++;
            }

            return enumValueToReturn;
        }

        private static int parseUserSelection(string i_inputAsString, int i_RangeOfEnum)
        {
            int userSelection;

            if (!int.TryParse(i_inputAsString, out userSelection))
            {
                throw new FormatException("Invalid Input");
            }

            if (userSelection < 1 || userSelection > i_RangeOfEnum)
            {
                throw new ValueOutOfRangeException(1, i_RangeOfEnum);
            }

            return userSelection;
        }

        private static void showMainMenu()
        {
            Console.WriteLine(k_MainMenu);
        }
    }
}