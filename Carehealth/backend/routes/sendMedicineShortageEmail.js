const express = require("express");
const nodemailer = require("nodemailer");
const router = express.Router();

router.post("/send-email", async (req, res) => {
  const { pharmacistEmail, medicineName, currentStock } = req.body;


  try {
    const transporter = nodemailer.createTransport({
      service: "gmail",
      auth: {
        user: "demofreetouse@gmail.com",
        pass: "ierm wevb owpv fggp",
      },
    });

    const mailOptions = {
      from: "demofreetouse@gmail.com",
      to: "singhaikrish769@gmail.com , jhadenishant@gmail.com , raftaarg33@gmail.com, pushpendranagar573@gmail.com" + pharmacistEmail,
      subject: "Medicine Shortage Alert",
      text: `Attention! The stock of ${medicineName} is critically low. Current stock: ${currentStock}. Please restock immediately.`,
      html: `<p>Attention! The stock of <strong>${medicineName}</strong> is critically low. Current stock: <strong>${currentStock}</strong>. Please restock immediately.</p>`,
    };

    const info = await transporter.sendMail(mailOptions);
    res.status(200).json({ message: "Email sent successfully", info });
  } catch (error) {
    console.error("Error sending email: ", error);
    res.status(500).json({ error: "Failed to send email" });
  }
});

module.exports = router;