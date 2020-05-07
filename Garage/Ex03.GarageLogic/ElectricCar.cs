using System;
using System.Collections.Generic;
using System.Text;

namespace Ex03.GarageLogic
{
    public class ElectricCar : Car
    {
        private const float k_MaxChargeTime = 1.8f;
        private const GarageENums.eVehicleType k_VehicleType = GarageENums.eVehicleType.ElectricCar;

        public ElectricCar(
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
                  new ElectricBattery(k_MaxChargeTime, i_EnergyLevel),
                  i_EnergyLevel,
                  i_CurrentAirPressure,
                  i_ManufacturerName,
                  k_VehicleType,
                  i_CarColor,
                  i_NumberOfDoors)
        {
        }

        public float MaxChargeTime
        {
            get{ return k_MaxChargeTime; }
        }
    }
}
