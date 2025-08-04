const express = require("express");
const router = express.Router();
const InventoryRequest = require("../Models/InventoryRequest");
const Demand = require("../Models/InventoryRequest");  // Path to your model

// Route to get all demands
router.get("/demands", async (req, res) => {
  try {
    // Fetch only demands where handled_by_pharmacist is true
    const demands = await InventoryRequest.find({ handled_by_pharmacist: false });
    res.status(200).json(demands);
  } catch (error) {
    console.error("Error fetching demands:", error);
    res.status(500).json({ message: "Error fetching demands" });
  }
});


router.get('/demands/:id', async (req, res) => {
  const { id } = req.params;
  try {
    const demand = await InventoryRequest.findById(id);
    if (!demand) {
      return res.status(404).json({ message: 'Demand not found' });
    }

    // Extract and format the requests
    const demandData = demand.requests.map(request => ({
      medicine_name: request.medicine_name,
      quantity: request.quantity,
    }));

    res.status(200).json(demandData);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to fetch demand details' });
  }
});


// Update demand status by ID
router.patch('/demands/:id', async (req, res) => {
  const { id } = req.params;
  const { handled_by_pharmacist, grand_total } = req.body;

  try {
    const demand = await InventoryRequest.findById(id);
    if (!demand) {
      return res.status(404).json({ message: 'Demand not found' });
    }

    demand.handled_by_pharmacist = handled_by_pharmacist;
    demand.grand_total = grand_total;
    await demand.save();

    res.status(200).json({ message: 'Demand updated successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Failed to update demand' });
  }
});

router.get('/hospital-bills', async (req, res) => {
  try {
    // Fetch bills with handled_by_pharmacist: true and select required fields
    const bills = await InventoryRequest.find(
      { handled_by_pharmacist: true },
      'doctorName department request_date grand_total'
    );
    console.log('Fetched bills:', bills); // Log fetched bills
    res.json(bills);
  } catch (error) {
    console.error('Error fetching hospital bills:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// Route to handle inventory requests
router.post("/save-inventory-request", async (req, res) => {
  try {
    const { doctorName, department, requests, grand_total } = req.body;

    // Validate required fields
    if (!doctorName || !department || !requests || requests.length === 0) {
      return res.status(400).json({ error: "All fields are required." });
    }

    // Create a new inventory request document
    const newRequest = new InventoryRequest({
      doctorName,
      department,
      requests,
      grand_total,
    });

    // Save to database
    await newRequest.save();
    res.status(201).json({ message: "Inventory request saved successfully!", newRequest });
  } catch (error) {
    console.error("Error saving inventory request:", error);
    res.status(500).json({ error: "Internal server error." });
  }
});



module.exports = router;
