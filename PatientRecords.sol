// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract PatientRecords {
    struct Patient {
        uint id;
        string name;
        string diagnosis;
        string treatmentPlan;
    }

    mapping(uint => Patient) public patients;
    uint public patientCount;

    event PatientAdded(uint id, string name, string diagnosis, string treatmentPlan);

    function addPatient(string memory _name, string memory _diagnosis, string memory _treatmentPlan) public {
        patientCount++;
        patients[patientCount] = Patient(patientCount, _name, _diagnosis, _treatmentPlan);
        emit PatientAdded(patientCount, _name, _diagnosis, _treatmentPlan);
    }

    function getPatient(uint _id) public view returns (uint, string memory, string memory, string memory) {
        Patient memory p = patients[_id];
        return (p.id, p.name, p.diagnosis, p.treatmentPlan);
    }
}
