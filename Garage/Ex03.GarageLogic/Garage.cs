using System;
using System.Collections.Generic;
using System.Text;

namespace Ex03.GarageLogic
{
    public class Garage
    {
        private readonly Dictionary<string, GarageVehicle> m_DictOfGarageVehicles;
        public readonly float k_MaxAirPressureCar = 31f;
        public readonly float k_MaxAirPressureMotorcycle = 33f;
        public readonly float k_MaxAirPressureTruck = 26f;
        public readonly float k_TankSizeFuelCar = 55f;
        public readonly float k_TankSizeFuelMotorcycle = 8f;
        public readonly float k_TankSizeFuelTruck = 110f;
        public readonly float k_BatteryTimeElectricityCar = 1.8f;
        public readonly float k_BatteryTimeElectricityMotorcycle = 1.4f;

        public Garage()
        {
            m_DictOfGarageVehicles = new Dictionary<string, GarageVehicle>();
        }

        public bool AddVehicleToGarage(Vehicle i_Vehicle)
        {
            if (i_Vehicle == null)
            {
                throw new NullReferenceException("Vehicle undefined, null reference received");
            }

            bool vehicleIsInGarage = m_DictOfGarageVehicles.ContainsKey(i_Vehicle.RegistrationPlate);
            GarageVehicle vehicleToAdd;

            if (vehicleIsInGarage)
            {
                ChangeVehicleStatus(i_Vehicle.RegistrationPlate, GarageENums.eVehicleStatus.InRepair);
            }
            else
            {
                vehicleToAdd = new GarageVehicle(i_Vehicle);
                m_DictOfGarageVehicles.Add(i_Vehicle.RegistrationPlate, vehicleToAdd);
            }

            return vehicleIsInGarage;
        }

        public void ChangeVehicleStatus(string i_PlateNumber, GarageENums.eVehicleStatus i_NewVehicleStatus)
        {
            GarageVehicle vehicleToChangeStatus = getVehicleFromDictionary(i_PlateNumber);
            if (vehicleToChangeStatus == null)
            {
                throw new ArgumentException(string.Format("Vehicle with registration plate number {0} not found ", i_PlateNumber));
            }

            vehicleToChangeStatus.VehicleStatusInGarage = i_NewVehicleStatus;
        }

        public void InflateVehicleWheelsToMax(string i_PlateNumber)
        {
            GarageVehicle vehicleToInflateWheels = getVehicleFromDictionary(i_PlateNumber);
            if (vehicleToInflateWheels == null)
            {
                throw new ArgumentException(string.Format("Vehicle with registration plate number {0} not found ", i_PlateNumber));
            }

            foreach (Wheel wheel in vehicleToInflateWheels.Vehicle.Wheels)
            {
                wheel.InflateWheelToMax();
            }
        }

        public void FuelOrChargeVehicle(string i_PlateNumber, Energy.eEnergyType i_EnergyType, float i_AmountToFill)
        {
            Energy.eEnergyType m_EnergyType = i_EnergyType;
            GarageVehicle vehicleToFillEnergy = getVehicleFromDictionary(i_PlateNumber);
            if (vehicleToFillEnergy == null)
            {
                throw new ArgumentException(string.Format("Vehicle with registration plate number {0} not found ", i_PlateNumber));
            }

            else if (vehicleToFillEnergy.Vehicle.EnergyType is ElectricBattery)
            {
                (vehicleToFillEnergy.Vehicle.EnergyType as ElectricBattery).Charge(i_AmountToFill);
            }
            else if (vehicleToFillEnergy.Vehicle.EnergyType is FuelTank)
            {
                FuelTank.eFuelType m_fuelType = (FuelTank.eFuelType)Enum.Parse(typeof(FuelTank.eFuelType), m_EnergyType.ToString());
                (vehicleToFillEnergy.Vehicle.EnergyType as FuelTank).FillTank(i_AmountToFill, m_fuelType);
            }
            else
            {
                throw new ArgumentException("Energy type is incompatible");
            }

        }

        public Energy GetVehicleEnergyType(string i_PlateNumber)
        {
            GarageVehicle vehicleToGetData = getVehicleFromDictionary(i_PlateNumber);
            return vehicleToGetData.EnergyType;
        }

        private GarageVehicle getVehicleFromDictionary(string i_PlateNumber)
        {
            GarageVehicle garageVehicle;
            bool vehicleExists = m_DictOfGarageVehicles.TryGetValue(i_PlateNumber, out garageVehicle);
            if (!vehicleExists)
            {
                garageVehicle = null;
            }

            return garageVehicle;
        }

        public string GetVehicleData(string i_RegistrationPlate)
        {
            GarageVehicle vehicleToGetData = getVehicleFromDictionary(i_RegistrationPlate);
            if (vehicleToGetData == null)
            {
                throw new ArgumentException(string.Format("Vehicle with registration plate number {0} not found ", i_RegistrationPlate));
            }

            return vehicleToGetData.ToString();
        }

        public List<string> GetRegistrationPlate()
        {
            List<string> plateNumbers = new List<string>();
            foreach (string plateNumber in m_DictOfGarageVehicles.Keys)
            {
                plateNumbers.Add(plateNumber);
            }

            return plateNumbers;
        }

        public List<string> GetRegistrationPlate(GarageENums.eVehicleStatus i_StatusInGarage)
        {
            List<string> plateNumbers = new List<string>();
            foreach (GarageVehicle vehicleInGarage in m_DictOfGarageVehicles.Values)
            {
                if (vehicleInGarage.VehicleStatusInGarage == i_StatusInGarage)
                {
                    plateNumbers.Add(vehicleInGarage.Vehicle.RegistrationPlate);
                }
            }

            return plateNumbers;
        }

        public float MaxAirPressureCar
        {
            get { return k_MaxAirPressureCar; }
        }

        public float MaxAirPressureMotorcycle
        {
            get { return k_MaxAirPressureMotorcycle; }
        }

        public float MaxAirPressureTruck
        {
            get { return k_MaxAirPressureTruck; }
        }

        public float TankSizeFuelCar
        {
            get { return k_TankSizeFuelCar; }
        }

        public float TankSizeFuelMotorcycle
        {
            get { return k_TankSizeFuelMotorcycle; }
        }

        public float TankSizeFuelTruck
        {
            get { return k_TankSizeFuelTruck; }
        }

        public float BatteryTimeElectricityCar
        {
            get { return k_BatteryTimeElectricityCar; }
        }

        public float BatteryTimeElectricityMotorcycle
        {
            get { return k_BatteryTimeElectricityMotorcycle; }
        }


        private class GarageVehicle
        {
            private GarageENums.eVehicleStatus m_VehicleStatusInGarage;
            private Vehicle m_Vehicle;

            public GarageVehicle(Vehicle i_Vehicle)
            {
                m_Vehicle = i_Vehicle;
                m_VehicleStatusInGarage = GarageENums.eVehicleStatus.InRepair;
            }

            public GarageENums.eVehicleStatus VehicleStatusInGarage
            {
                get { return m_VehicleStatusInGarage; }
                set { m_VehicleStatusInGarage = value; }
            }

            public Energy EnergyType
            {
                get { return m_Vehicle.EnergyType; }
            }

            public Vehicle Vehicle
            {
                get { return m_Vehicle; }
            }

            public override string ToString()
            {
                string vehicleInfo = string.Format(
                                                    "Status in garage: {1}{0}",
                                                    Environment.NewLine,
                                                    VehicleStatusInGarage);

                return Vehicle.ToString() + vehicleInfo;
            }
        }
    }
}
