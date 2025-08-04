// routes/BedManagement.js
const express = require("express");
const mongoose = require('mongoose')
const router = express.Router();
const Bed = require("../Models/Bed");
const BedPrice = require("../Models/BedPrice");
const Application = require("../Models/BedApplication");


// Apply for a bed
router.post("/apply-bed", async (req, res) => {
  try {
    const { name, contact, department, bedType, doctorId, doctorname } = req.body;

    // Check if doctorId is provided
    if (!doctorId) {
      return res.status(400).json({ message: "Doctor ID is required" });
    }

    // Find a free bed of the requested type
    const freeBed = await Bed.findOne({ type: bedType, status: "Free" });

    if (!freeBed) {
      return res.status(400).json({ message: `No ${bedType} bed available` });
    }

    // Update bed status to "Occupied"
    freeBed.status = "Occupied";
    await freeBed.save();

    // Create and save application with doctor details
    const application = new Application({
      name,
      contact,
      department,
      bedType,
      doctorId,
      doctorname, // Store doctor name
      bedNumber: freeBed.bedNumber,
    });

    await application.save();

    res.json({ message: "Bed allocated successfully", bedNumber: freeBed.bedNumber });
  } catch (error) {
    console.error("Error allocating bed:", error);
    res.status(500).json({ message: "Error allocating bed" });
  }
});

router.get("/bed-status-admin" , async (req,res) => {
    try{
      const applications = await Application.find();
      res.json(applications);
    }catch(err){
      console.error("Error fetching bed status:", err);
      res.status(500).json({ message: "Error fetching bed status" });
    }
});


// Get all bed applications
router.get("/bed-status-doctor/:doctorId", async (req, res) => {
  try {
  
    const { doctorId } = req.params; // Extract doctorId from URL params

    if (!doctorId) {
      return res.status(400).json({ message: "doctorId is required" });
    }

    if (!mongoose.Types.ObjectId.isValid(doctorId)) {
      return res.status(400).json({ message: "Invalid doctor ID format" });
    }

    // Fetch applications assigned to the doctor
    const applications = await Application.find({ doctorId }).sort({ allocationTime: -1 });

    if (!applications || applications.length === 0) {
      console.warn("No applications found for this doctor.");
      return res.status(200).json([]); // Return empty array if no applications found
    }

    console.log("Fetched Applications:", applications);
    res.json(applications); // Send the fetched applications as a response
  } catch (error) {
    console.error("Error fetching bed applications:", error);
    res.status(500).json({ message: "Error fetching bed applications", error });
  }
});


// Add new beds
router.post("/add-beds", async (req, res) => {
  try {
    const { type, count } = req.body;
    const lastBed = await Bed.findOne({ type }).sort({ number: -1 });
    let lastNumber = lastBed ? lastBed.number : 0;
    const prefix = type === "ICU" ? "ICU" : type === "Private" ? "PVT" : "GN";

    for (let i = 1; i <= count; i++) {
      const newNumber = lastNumber + i;
      const bedNumber = `${prefix}B${newNumber}`;
      await Bed.create({ bedNumber, number: newNumber, type, status: "Free" });
    }

    res.json({ message: `${count} ${type} beds added successfully` });
  } catch (error) {
    res.status(500).json({ message: "Error adding beds" });
  }
});

// Remove a bed
router.delete("/remove-bed/:bedNumber", async (req, res) => {
  try {
    const { bedNumber } = req.params;
    const bed = await Bed.findOne({ bedNumber });

    if (!bed) {
      return res.status(404).json({ message: "Bed not found" });
    }
    if (bed.status === "Occupied") {
      return res.status(400).json({ message: "Cannot remove an occupied bed" });
    }
    await Bed.deleteOne({ bedNumber });
    res.json({ message: `Bed ${bedNumber} removed successfully` });
  } catch (error) {
    res.status(500).json({ message: "Error removing bed" });
  }
});

// Get free beds
router.get("/free-beds", async (req, res) => {
  try {
    const freeBeds = await Bed.find({ status: "Free" }).sort({ bedType: 1 });
    res.json(freeBeds);
  } catch (err) {
    res.status(500).json({ message: "Error fetching free beds", error: err.message });
  }
});

// Get bed statistics
router.get("/bed-stats", async (req, res) => {
  const types = ["ICU", "Private", "General"];
  const stats = {};

  for (const type of types) {
    const total = await Bed.countDocuments({ type });
    const available = await Bed.countDocuments({ type, status: "Free" });
    stats[type] = { total, available };
  }

  res.json(stats);
});

// Get occupied beds with prices
router.get("/occupied-beds", async (req, res) => {
  try {
    const occupiedBeds = await Bed.find({ status: "Occupied" });
    const bedsWithPrices = await Promise.all(
      occupiedBeds.map(async (bed) => {
        const bedPrice = await BedPrice.findOne({ bedNumber: bed.bedNumber });
        
        return {
          bedNumber: bed.bedNumber,
          type: bed.type,
          pricePerDay: bedPrice ? bedPrice.pricePerDay : defaultPrices[bed.type],
        };
      })
    );
    res.json(bedsWithPrices);
  } catch (err) {
    res.status(500).json({ message: "Error fetching occupied beds", error: err.message });
  }
});

// Get default bed prices
router.get("/get-default-prices", async (req, res) => {
  const defaultPrices = await BedPrice.find();
  console.log(defaultPrices);
  res.json(defaultPrices);
  console.log(defaultPrices);
});

// Update default bed prices
router.put("/update-default-prices", async (req, res) => {
  try {
    const { ICU, Private, General } = req.body;
    console.log(ICU, Private, General);

    const updates = await Promise.all([
      BedPrice.findOneAndUpdate({ bedType: "ICU" }, { pricePerDay: ICU }, { new: true }),
      BedPrice.findOneAndUpdate({ bedType: "Private" }, { pricePerDay: Private }, { new: true }),
      BedPrice.findOneAndUpdate({ bedType: "General" }, { pricePerDay: General }, { new: true })
    ]);
    console.log(updates);

    res.json({ message: "Default prices updated successfully", updates });
  } catch (error) {
    console.error("Error updating prices:", error);
    res.status(500).json({ error: "Error updating default prices" });
  }
});


// ✅ 1️⃣ Generate Discharge Bill
router.post("/discharge-bill/:id", async (req, res) => {
  try {
    const { id } = req.params;
    console.log(`Generating bill for patient ID: ${id}`);

    // Find the application
    const application = await Application.findById(id);
    if (!application) {
      return res.status(404).json({ message: "Application not found" });
    }

    // Calculate Days Stayed
    const allocationTime = new Date(application.allocationTime || application.createdAt);
    const currentTime = new Date();
    const daysStayed = Math.max(1, Math.ceil((currentTime - allocationTime) / (1000 * 60 * 60 * 24)));

    // Get Bed Price
    const bedPrice = await BedPrice.findOne({ bedType: application.bedType });
    const pricePerDay = bedPrice ? bedPrice.pricePerDay : 0;
    const totalCost = daysStayed * pricePerDay;

    // Create and Save Bill
    const dischargeBill = new DischargeBill({
      name: application.name,
      contact: application.contact,
      department: application.department,
      bedType: application.bedType,
      bedNumber: application.bedNumber,
      allocationTime,
      daysStayed,
      totalCost,
    });

    await dischargeBill.save();
    console.log("Bill Saved:", dischargeBill);

    // Mark Bed as Free
    await Bed.updateOne({ bedNumber: application.bedNumber }, { status: "Free" });

    res.json({ message: "Bill generated and saved successfully", dischargeBill });
  } catch (error) {
    console.error("Error generating discharge bill:", error);
    res.status(500).json({ message: "Error generating discharge bill", error: error.message });
  }
});

// ✅ 2️⃣ Discharge a Patient
router.put("/discharge/:id", async (req, res) => {
  try {
    const { id } = req.params;
    console.log(`Discharging patient with ID: ${id}`);

    // Find Application
    const application = await Application.findById(id);
    if (!application) {
      return res.status(404).json({ message: "Application not found" });
    }

    console.log("Application found:", application);

    // Find and Update Bed Status
    const bed = await Bed.findOne({ bedNumber: application.bedNumber });
    if (!bed) {
      return res.status(404).json({ message: "Bed not found" });
    }

    console.log("Bed found:", bed);
    bed.status = "Free"; // or bed.isOccupied = false;
    await bed.save();

    // Delete the Application
    await Application.findByIdAndDelete(id);

    res.json({ message: "Patient discharged successfully" });
  } catch (error) {
    console.error("Error discharging patient:", error);
    res.status(500).json({ message: "Internal Server Error", error: error.message });
  }
});

// ✅ 3️⃣ Get All Discharge Bills
router.get("/discharge-bill", async (req, res) => {
  try {
    const bills = await DischargeBill.find().sort({ allocationTime: -1 });
    res.json(bills);
  } catch (error) {
    console.error("Error fetching all bills:", error);
    res.status(500).json({ message: "Internal Server Error", error: error.message });
  }
});

router.get("/totalbed", async (req, res) => {
  try {
    const totalBeds = await Bed.countDocuments();
    const freeBeds = await Bed.countDocuments({ status: "Free" });
    const occupiedBeds = await Bed.countDocuments({ status: "Occupied" });

    console.log("Total Beds:", totalBeds);
    console.log("Free Beds:", freeBeds);
    console.log("Occupied Beds:", occupiedBeds);
    res.json({ totalBeds, freeBeds, occupiedBeds });
  } catch (error) {
    console.error("Error fetching bed counts:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});




module.exports = router;