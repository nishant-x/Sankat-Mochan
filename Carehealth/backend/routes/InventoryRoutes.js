const express = require("express");
const router = express.Router();

const Inventory = require('../Models/Inventory');
const PatientPrescriptions = require('../Models/Patientprescription');
const InventoryManager = require('../Models/InventoryManagerRegistration');


// ################# Inventory Routes ###############################
router.get('/api/inventory', async (req, res) => {
    try {
        const inventory = await Inventory.find().sort({ expiry_date: 1 });
        res.json(inventory);
        // console.log('inventory', inventory)
    } catch (error) {
        res.status(500).json({ message: 'Failed to fetch inventory' });
    }
});
//search item from  inventory 
router.get('/search/inventory', async (req, res) => {
    try {
        const search = req.query.search || "";
        const inventory = await Inventory.find({
            name: { $regex: search, $options: "i" }, // Case-insensitive regex search
        })
        res.json(inventory);
    } catch (error) {
        console.error("Error fetching inventory:", error);
        res.status(500).json({ message: "Failed to fetch inventory" });
    }
});


// fetch item on the basis of id
router.get('/api/inventory/:id', async (req, res) => {
    try {
        const item = await Inventory.findById(req.params.id);
        if (!item) return res.status(404).json({ message: "Product not found" });
        res.json(item);
    } catch (error) {
        res.status(500).json({ message: "Server error" });
    }
});


//update quantity
router.patch('/api/inventory/:id', async (req, res) => {
    const { id } = req.params;
    const { quantity } = req.body;

    try {
        const updatedItem = await Inventory.findByIdAndUpdate(id, { quantity }, { new: true });
        if (!updatedItem) {
            return res.status(404).json({ message: 'Item not found' });
        }
        res.json(updatedItem);
    } catch (error) {
        console.error('Error updating inventory:', error.message);
        res.status(500).json({ message: 'Failed to update inventory', error: error.message });
    }
});

// add new item
router.post('/api/inventory', async (req, res) => {
    const { name, quantity, manufacturer, expiry_date, manufacturing_date, cost } = req.body;

    if (!name || !quantity || quantity <= 0 || !cost || cost <= 0 || !manufacturing_date) {
        console.log('Invalid input values');
        return res.status(400).json({ message: 'Invalid input values' });
    }

    try {
        // Check if the item already exists
        const existingItem = await Inventory.findOne({ name });

        if (existingItem) {
            return res.status(400).json({ message: 'Item already exists in the inventory' });
        }

        const newItem = new Inventory({
            name,
            quantity,
            manufacturer,
            expiry_date,
            manufacturing_date,
            cost,
        });

        await newItem.save();
        res.status(201).json(newItem);
    } catch (error) {
        console.error('Error adding new inventory item:', error);
        res.status(500).json({ message: 'Failed to add inventory item' });
    }
});

// API to check if an item already exists in the inventory
// Check if name query is present
router.get('/api/inventory/check', async (req, res) => {
    const { name } = req.query;

    if (!name) {
        console.log('No item name provided in query');
        return res.status(400).json({ message: 'Item name is required' });
    }

    try {
        console.log('Checking inventory for item:', name);

        // Count how many items exist with the same name
        const count = await Inventory.countDocuments({ name });

        if (count > 0) {
            return res.status(200).json({ exists: true, count });
        } else {
            return res.status(200).json({ exists: false });
        }
    } catch (error) {
        console.error('Error checking inventory item:', error);
        return res.status(500).json({ message: 'Failed to check item in inventory' });
    }
});

router.get('/api/expired-products', async (req, res) => {
    try {
        const expiredItems = await Inventory.find({ expiry_date: { $lt: new Date() } });
        if (expiredItems.length > 0) res.json(expiredItems);
        else res.status(404).json({ message: 'No expired products found' });
    } catch (error) {
        res.status(500).json({ message: 'Server Error' });
    }
});

router.get('/api/about_to_expire', async (req, res) => {
    try {
        const twoMonthsLater = new Date();
        twoMonthsLater.setMonth(twoMonthsLater.getMonth() + 2);
        const aboutToExpireItems = await Inventory.find({ expiry_date: { $gte: new Date(), $lte: twoMonthsLater } }).sort({ expiry_date: 1 });
        res.json(aboutToExpireItems);
    } catch (err) {
        res.status(500).json({ error: 'Failed to fetch items about to expire' });
    }
});

router.get('/api/expired-products/:id', async (req, res) => {
    try {
        const product = await Inventory.findById(req.params.id);
        if (!product) return res.status(404).json({ message: 'Product not found' });
        res.json(product);
    } catch (error) {
        res.status(500).json({ message: 'Server Error' });
    }
});

router.patch('/api/expired-products/:id', async (req, res) => {
    const { expiry_date } = req.body;
    if (!expiry_date) return res.status(400).json({ message: 'Invalid expiry date' });

    try {
        const updatedProduct = await Inventory.findByIdAndUpdate(req.params.id, { expiry_date }, { new: true });
        if (!updatedProduct) return res.status(404).json({ message: 'Product not found' });
        res.json(updatedProduct);
    } catch (error) {
        res.status(500).json({ message: 'Failed to update product' });
    }
});

// Patient Routes
router.get('/api/patients', async (req, res) => {
    try {
        const patients = await PatientPrescriptions.find({ handled_by_pharmacist: false }).sort({ prescri_date: -1 });
        res.json(patients);
    } catch (error) {
        res.status(500).json({ message: 'Failed to fetch patients' });  
    }
});

router.get('/api/patients/:id', async (req, res) => {
    try {
        const patient = await PatientPrescriptions.findById(req.params.id);
        if (!patient) return res.status(404).json({ message: 'Patient not found' });
        res.json(patient);
    } catch (error) {
        res.status(500).json({ message: 'Failed to fetch patient' });
    }
});

router.patch('/api/patients/:id', async (req, res) => {
    const { id } = req.params;
    const { handled_by_pharmacist, grand_total } = req.body;

    try {
        const updateFields = { handled_by_pharmacist };

        // Include grand_total in the update if it's provided
        if (grand_total !== undefined) {
            updateFields.grand_total = grand_total;
        }

        const updatedPatient = await PatientPrescriptions.findByIdAndUpdate(id, updateFields, { new: true });

        if (!updatedPatient) {
            return res.status(404).json({ message: 'Patient not found' });
        }

        res.json(updatedPatient);
    } catch (error) {
        console.error('Error updating patient status:', error.message);
        res.status(500).json({ message: 'Failed to update patient status', error: error.message });
    }
});

module.exports = router;
