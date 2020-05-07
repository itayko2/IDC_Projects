using System;
using System.Collections.Generic;
using System.Text;

namespace Ex03.GarageLogic
{
    public class FuelCar : Car
    {
        private const FuelTank.eFuelType k_FuelType = FuelTank.eFuelType.Octan96;
        private const float k_FuelTankCapacity = 55f;
        private const GarageENums.eVehicleType k_VehicleType = GarageENums.eVehicleType.FuelCar;

        public FuelCar(
            string i_OwnersName,
            string i_OwnersPhone,
            string i_Model,
            string i_RegistrationPlate,
            float i_EnergyLevel,
            float i_CurrentAirPressure,
            string i_ManufacturerName,
            Car.eCarColor i_CarColor,
            Car.eNumberOfCarDoors i_NumberOfDoors)
            : base(
                  i_OwnersName,
                  i_OwnersPhone,
                  i_Model,
                  i_RegistrationPlate,
                  new FuelTank(k_FuelTankCapacity, i_EnergyLevel, k_FuelType),
                  i_EnergyLevel,
                  i_CurrentAirPressure,
                  i_ManufacturerName,
                  k_VehicleType,
                  i_CarColor,
                  i_NumberOfDoors)
        {
        }

        public float FuelTankCapacity
        {
            get { return k_FuelTankCapacity; }
        }

        public FuelTank.eFuelType FuelType
        {
            get { return k_FuelType; }
        }
    }
}
