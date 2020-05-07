using System;
using System.Collections.Generic;
using System.Text;

namespace Ex03.GarageLogic
{
    public static class VehicleGenerator
    {
        public static Vehicle GenerateVehicle(GarageENums.eVehicleType i_VehicleType, List<object> i_VehicleInfo)
        {
            Vehicle newVehicle = null;

            switch (i_VehicleType)
            {
                case GarageENums.eVehicleType.FuelCar:
                    newVehicle = new FuelCar((string)i_VehicleInfo[0],
                                             (string)i_VehicleInfo[1],
                                             (string)i_VehicleInfo[2],
                                             (string)i_VehicleInfo[3],
                                             (float)i_VehicleInfo[4],
                                             (float)i_VehicleInfo[5],
                                             (string)i_VehicleInfo[6],
                                             (Car.eCarColor)i_VehicleInfo[7],
                                             (Car.eNumberOfCarDoors)i_VehicleInfo[8]);
                    break;
                case GarageENums.eVehicleType.ElectricCar:
                    newVehicle = new ElectricCar((string)i_VehicleInfo[0],
                                             (string)i_VehicleInfo[1],
                                             (string)i_VehicleInfo[2],
                                             (string)i_VehicleInfo[3],
                                             (float)i_VehicleInfo[4],
                                             (float)i_VehicleInfo[5],
                                             (string)i_VehicleInfo[6],
                                             (Car.eCarColor)i_VehicleInfo[7],
                                             (Car.eNumberOfCarDoors)i_VehicleInfo[8]);
                    break;
                case GarageENums.eVehicleType.FuelMotorcycle:
                    newVehicle = new FuelMotorcycle((string)i_VehicleInfo[0],
                                             (string)i_VehicleInfo[1],
                                             (string)i_VehicleInfo[2],
                                             (string)i_VehicleInfo[3],
                                             (float)i_VehicleInfo[4],
                                             (float)i_VehicleInfo[5],
                                             (string)i_VehicleInfo[6],
                                             (Motorcycle.eMotorcycleLicenseType)i_VehicleInfo[7],
                                             (int)i_VehicleInfo[8]);
                    break;
                case GarageENums.eVehicleType.ElectricMotorcycle:
                    newVehicle = new ElectricMotorcycle((string)i_VehicleInfo[0],
                                             (string)i_VehicleInfo[1],
                                             (string)i_VehicleInfo[2],
                                             (string)i_VehicleInfo[3],
                                             (float)i_VehicleInfo[4],
                                             (float)i_VehicleInfo[5],
                                             (string)i_VehicleInfo[6],
                                             (Motorcycle.eMotorcycleLicenseType)i_VehicleInfo[7],
                                             (int)i_VehicleInfo[8]);
                    break;
                case GarageENums.eVehicleType.FuelTruck:
                    newVehicle = new FuelTruck((string)i_VehicleInfo[0],
                                             (string)i_VehicleInfo[1],
                                             (string)i_VehicleInfo[2],
                                             (string)i_VehicleInfo[3],
                                             (float)i_VehicleInfo[4],
                                             (float)i_VehicleInfo[5],
                                             (string)i_VehicleInfo[6],
                                             (float)i_VehicleInfo[7],
                                             (bool)i_VehicleInfo[8]);
                    break;
            }

            return newVehicle;
        }

    }
}
