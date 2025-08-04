const express = require("express");
const router = express.Router();
const Patient = require("../Models/PatientRegistration");
const Doctor = require("../Models/DoctorRegistration");
const InventoryManager = require("../Models/InventoryManagerRegistration");
const Admin = require("../Models/AdminRegistration");

const bcypt = require("bcrypt");
const jwt = require("jsonwebtoken");

router.post("/patient_register", (req, res) => {
  try {
    let {
      name,
      age,
      gender,
      contact,
      email,
      password,
      confirm_password,
      bloodGroup,
      address,
    } = req.body;

    bcypt.genSalt(10, (err, salt) => {
      bcypt.hash(password, salt, async (err, hash) => {
        if (password === confirm_password) {
          const patientRegistered = new Patient({
            name,
            age,
            gender,
            contact,
            email,
            password: hash,
            confirm_password: hash,
            bloodGroup,
            address,
          });

          const registered = await patientRegistered.save();
          console.log("Patient registered: ", registered);
          res.status(201).send("Patient registered successfully");
        } else {
          res.status(400).send("Passwords do not match");
        }
      });
    });
  } catch (error) {
    res.status(400).send(error);
  }
});

// Route to handle doctor registration form submissions
router.post("/doctor_register", async (req, res) => {
  try {
    let {
      name,
      age,
      gender,
      contact,
      email,
      password,
      confirm_password,
      specialization,
      experience,
      qualification,
    } = req.body;
    bcypt.genSalt(10, (err, salt) => {
      bcypt.hash(password, salt, async (err, hash) => {
        const newDoctor = new Doctor({
          name,
          age,
          gender,
          contact,
          email,
          password:hash,
          confirm_password:hash,
          specialization,
          experience,
          qualification,
        });

        const registeredDoctor = await newDoctor.save();
        res.status(201).send("Doctor registered successfully");
      });
    });
  } catch (err) {
    res.status(400).send(`Error: ${err.message}`);
  }
});

// Route to handle inventory manager registration form submissions
router.post("/inventory_register", async (req, res) => {
  try {
    let {
      name,
      age,
      gender,
      contact,
      email,
      password,
      confirm_password,
      department,
      yearsOfExperience,
    } = req.body;
    bcypt.genSalt(10, (err, salt) => {
      bcypt.hash(password, salt, async (err, hash) => {
        const newInventoryManager = new InventoryManager({
          name,
          age,
          gender,
          contact,
          email,
          password:hash,
          confirm_password:hash,
          department,
          yearsOfExperience,
        });

        const registeredInventoryManager = await newInventoryManager.save();
        res.status(201).send("Inventory Manager registered successfully");
      });
    });
  } catch (err) {
    res.status(400).send(`Error: ${err.message}`);
  }
});
// Route to handle admin registration form sumission
router.post("/admin_register", async (req, res) => {
  try {
    let { name, contact, email, password, confirm_password, userRole } =
      req.body;
    bcypt.genSalt(10, (err, salt) => {
      bcypt.hash(password, salt, async (err, hash) => {
        const newDoctor = new Admin({
          name,
          contact,
          email,
          password:hash,
          confirm_password:hash,
          userRole,
        });

        const registeredDoctor = await newDoctor.save();
        res.status(201).send("Doctor registered successfully");
      });
    });
  } catch (err) {
    res.status(400).send(`Error: ${err.message}`);
  }
});

router.post("/logout_token", async (req, res) => {
res.cookie("token","")
})

module.exports = router;
