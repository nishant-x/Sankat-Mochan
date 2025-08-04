const express = require("express");
const router = express.Router();
const Patient= require("../Models/PatientRegistration");
const doctor = require("../Models/DoctorRegistration");

router.get("/totalpatients", async(req,res)=>{
    try{
        const total = await Patient.find();
        res.json(total);
    }
    catch{
        res.status(500).json({ message: 'Error fetching appointments' });
    }
});

router.get("/totaldoctors", async(req,res)=>{
    try{
        const total = await doctor.find();
        res.json(total);
    }
    catch{
        res.status(500).json({ message: 'Error fetching appointments' });
    }
});


module.exports= router;