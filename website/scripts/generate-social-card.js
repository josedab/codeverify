/**
 * Generate PNG social card from SVG
 * 
 * This script converts the SVG social card to PNG format for
 * better compatibility with social media platforms that don't
 * support SVG in Open Graph images.
 */

const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

const INPUT_SVG = path.join(__dirname, '../static/img/social-card.svg');
const OUTPUT_PNG = path.join(__dirname, '../static/img/social-card.png');

async function generateSocialCard() {
  console.log('🖼️  Generating social card PNG...');
  
  try {
    // Read the SVG file
    const svgBuffer = fs.readFileSync(INPUT_SVG);
    
    // Convert SVG to PNG at 1200x630 (standard OG image size)
    await sharp(svgBuffer)
      .resize(1200, 630)
      .png({ quality: 90, compressionLevel: 9 })
      .toFile(OUTPUT_PNG);
    
    const stats = fs.statSync(OUTPUT_PNG);
    console.log(`✅ Generated ${OUTPUT_PNG} (${Math.round(stats.size / 1024)}KB)`);
  } catch (error) {
    console.error('❌ Failed to generate social card:', error.message);
    // Don't fail the build if sharp isn't available
    if (error.code === 'MODULE_NOT_FOUND') {
      console.log('ℹ️  Install sharp with: npm install --save-dev sharp');
      console.log('ℹ️  Skipping PNG generation, using SVG fallback');
    }
    process.exit(0); // Don't fail the build
  }
}

generateSocialCard();
