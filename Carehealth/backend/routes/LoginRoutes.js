const fs = require("fs");
const path = require("path");
const express = require("express");
const router = express.Router();
const Prescription = require("../Models/Patientprescription");
const Patient = require("../Models/PatientRegistration");
const InventoryManager = require("../Models/InventoryManagerRegistration");
const Doctor = require("../Models/DoctorRegistration");
const Admin = require("../Models/AdminRegistration");
const Appoinments = require("../Models/AppointmentSchema")
const jwt = require("jsonwebtoken");
const bcrypt = require("bcrypt");

const Chatbot = require("../Models/Chatbot");

const CHATBOT_DATA_DIR = path.join(__dirname, "../ChatBotData");

// Ensure the ChatBotData folder exists
if (!fs.existsSync(CHATBOT_DATA_DIR)) {
  fs.mkdirSync(CHATBOT_DATA_DIR, { recursive: true });
}

// Ensure the ChatBotData folder exists
if (!fs.existsSync(CHATBOT_DATA_DIR)) {
  fs.mkdirSync(CHATBOT_DATA_DIR, { recursive: true });
}

router.post("/patient_login", async (req, res) => {
  try {
    const { email, password } = req.body;

    // Check if patient exists
    const patient = await Patient.findOne({ email });
    if (!patient) {
      return res.status(404).json({ message: "User not found" });
    }

    // Compare hashed password
    const isMatch = await bcrypt.compare(password, patient.password);
    if (!isMatch) {
      return res.status(401).json({ message: "Invalid password" });
    }

    // Generate JWT token
    const token = jwt.sign(
      { id: patient._id, email: patient.email },
      "secretkey",
      { expiresIn: "1h" }
    );

    // Set token in HTTP-only cookie
    res.cookie("token", token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "strict",
      maxAge: 60 * 60 * 1000, // 1 hour
    });

    // Fetch prescriptions for the patient
    const prescriptions = await Prescription.find({ patient_id: patient._id });

    // Fetch all chatbot report details for the patient
    const chatbotData = await Chatbot.find({ patientId: patient._id });

    // Fetch all appointments of the patient
    const appointments = await Appoinments.find({ patientId: patient._id });
    console.log(`Fetched ${appointments.length} appointments for patient ${patient._id}`);

    // Define file path: ChatBotData/{patient_id}.txt
    const filePath = path.join(CHATBOT_DATA_DIR, `${patient._id}.txt`);

    // Prepare content for the file
    let content = `# Patient Details:
Name: ${patient.name}
Age: ${patient.age}
Gender: ${patient.gender}
Contact Number: ${patient.contact}
Blood Group: ${patient.bloodGroup}
Address: ${patient.address}
Email: ${patient.email}
Login Time: ${new Date().toISOString()}\n\n`;

    // Add prescription details
    content += "# Patient Prescriptions:\n";
    if (prescriptions.length > 0) {
      prescriptions.forEach((prescription, index) => {
        content += ` ${index + 1}. Prescription Details: ${prescription.prescription
          .map(
            (med) => `${med.medicine_name} (${med.dosage}, ${med.quantity} units)`
          )
          .join(", ")}\n`;
      });
    } else {
      content += "  No prescriptions found.\n";
    }

    // Add appointment details
    content += "\n# Patient Appointments:\n";
    if (appointments.length > 0) {
      appointments.forEach((appointment, index) => {
        content += ` Appointment ${index + 1}: Date: ${appointment.date}, Time: ${appointment.time}, Doctor: ${appointment.doctorName}, Department: ${appointment.department}\n`;
      });
    } else {
      content += "  No appointments found.\n";
    }

    // Add chatbot report details
    content += "\n# Patient Report Details:\n";
    if (chatbotData.length > 0) {
      chatbotData.forEach((chat, index) => {
        content += ` Report ${index + 1}. ${chat.reportDetails}\n`;
      });
    } else {
      content += "  No report details found.\n";
    }

    // Write content to the text file
    fs.writeFileSync(filePath, content, "utf8");

    res.json({ patient });
  } catch (err) {
    console.error("Error:", err);
    res.status(500).send(`Error: ${err.message}`);
  }
});



router.get("/fetchtxtfile/:patientId", async (req, res) => {
  try {
    const { patientId } = req.params;
    const filePath = path.join(`CHATBOT_DATA_DIR, ${patientId}.txt`);

    // Check if the file exists
    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ message: "File not found" });
    }

    // Read the file and send it as a response
    const fileContent = fs.readFileSync(filePath, "utf8");
    res.setHeader("Content-Type", "text/plain");
    res.send(fileContent);
  } catch (err) {
    console.error("Error fetching file:", err);
    res.status(500).send(`Error: ${err.message}`);
  }
});

router.post("/doctor_login", async (req, res) => {
  try {
    const { email, password } = req.body;

    // Check if doctor exists
    const doctor = await Doctor.findOne({ email });
    if (!doctor) {
      return res.status(404).json({ message: "User not found" });
    }

    // Compare passwords
    const isMatch = await bcrypt.compare(password, doctor.password);
    if (!isMatch) {
      return res.status(401).json({ message: "Invalid password" });
    }

    // Generate JWT token
    const token = jwt.sign({ email: doctor.email, id: doctor._id }, "secretkey", { expiresIn: "1h" });

    // Send token in HTTP-Only cookie
    res.cookie("token", token, {
      httpOnly: true, // Prevents XSS attacks
      secure: process.env.NODE_ENV === "production", // Secure only in production
      sameSite: "strict", // Helps prevent CSRF attacks
      maxAge: 60 * 60 * 1000, // 1 hour expiration
    });

    // Send response
    res.status(200).json({ doctor, token });
  } catch (err) {
    console.error("Login Error:", err);
    res.status(500).json({ message: "Server error. Please try again later." });
  }
});


// Route to handle inventory  login
router.post("/inventory_manager_login", async (req, res) => {
    try {
      const { email, password } = req.body;
  
      // Check if inventory manager exists
      const inventoryManager = await InventoryManager.findOne({ email });
      if (!inventoryManager) {
        return res.status(404).json({ message: "User not found" });
      }
  
      // Compare passwords securely
      const isMatch = await bcrypt.compare(password, inventoryManager.password);
      if (!isMatch) {
        return res.status(401).json({ message: "Invalid password" });
      }
  
      // Generate JWT token
      const token = jwt.sign({ email: inventoryManager.email, id: inventoryManager._id }, "secretkey", { expiresIn: "1h" });
  
      // Set token in HTTP-only cookie
      res.cookie("token", token, {
        httpOnly: true, // Prevents XSS attacks
        secure: process.env.NODE_ENV === "production", // Secure in production
        sameSite: "strict", // Prevents CSRF attacks
        maxAge: 60 * 60 * 1000, // Token valid for 1 hour
      });
  
      // Send response
      res.status(200).json({ inventoryManager, token });
    } catch (err) {
      console.error("Login Error:", err);
      res.status(500).json({ message: "Server error. Please try again later." });
    }
  });
// Route to handle admin login

router.post("/admin_login", async (req, res) => {
    try {
      const { email, password } = req.body;
  
      // Check if admin exists
      const admin = await Admin.findOne({ email });
      if (!admin) {
        return res.status(404).json({ message: "User not found" });
      }
  
      // Compare hashed password
      const isMatch = await bcrypt.compare(password, admin.password);
      if (!isMatch) {
        return res.status(401).json({ message: "Invalid password" });
      }
  
      // Generate JWT token
      const token = jwt.sign(
        { id: admin._id, email: admin.email, role: admin.userRole },
        "secretkey",
        { expiresIn: "1h" }
      );
  
      // Set token in HTTP-only cookie
      res.cookie("token", token, {
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "strict",
        maxAge: 60 * 60 * 1000, // 1 hour
      });
  
      // Return admin details
      res.status(200).json({
        id: admin._id,
        email: admin.email,
        userRole: admin.userRole,
        token,
      });
    } catch (err) {
      console.error("Admin Login Error:", err);
      res.status(500).json({ message: "Server error. Please try again later." });
    }
  });
module.exports = router;