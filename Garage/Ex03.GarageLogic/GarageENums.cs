using System;
namespace Ex03.GarageLogic
{
    public class GarageENums
    {
        public enum eVehicleStatus
        {
            InRepair,
            Repaired,
            Paid
        }

        public enum eVehicleType
        {
            FuelCar = 1,
            ElectricCar = 2,
            FuelMotorcycle = 3,
            ElectricMotorcycle = 4,
            FuelTruck = 5
        }
    }
}
