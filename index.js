// Import von externen Bibliotheken
import express from "express";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";
import Replicate from "replicate";
import { SpeechClient } from "@google-cloud/speech";
import multer from "multer";
import ffmpeg from "fluent-ffmpeg";
import fs from "fs";
import { readFile } from "fs/promises";

import dotenv from "dotenv";
dotenv.config();
import Groq from "groq-sdk";

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

const app = express();
app.use(express.json());
const PORT = process.env.PORT || 4000;

const client = new SpeechClient({
  keyFilename: "./studybuddy-419317-76c369013539.json",
});
const replicate = new Replicate({ auth: process.env.REPLICATE_API_KEY });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

const upload = multer();
ffmpeg.setFfmpegPath("C:/ffmpeg/bin/ffmpeg.exe");
ffmpeg.setFfprobePath("C:/ffmpeg/bin/ffprobe.exe");

app.listen(PORT, () => console.log(`Server is running on port ${PORT}`));

app.post("/speech-to-text", upload.single("file"), async (req, res) => {
  const userId = "user";

  if (!req.file) {
    return res.status(400).send("Keine Datei hochgeladen.");
  }

  const tempInputPath = `${Date.now()}_original_audio`;
  const tempOutputPath = `${Date.now()}_converted_audio.wav`;
  fs.writeFileSync(tempInputPath, req.file.buffer);

  try {
    // Konvertierung der Audiodatei
    await convertAudio(tempInputPath, tempOutputPath);
    // Transkription der Audiodatei
    const transcriptionText = await transcribeAudio(tempOutputPath);
    // Holen der bisherigen Konversation oder erstellen einer neuen mit dem System-Prompt
    let conversation = await readConversationFile(userId);
    if (conversation.length === 0) {
      // Füge System-Prompt hinzu, wenn es die erste Anfrage ist
      conversation.push({
        role: "system",
        content:
          "You are a AI assistant that helps students with their studies. You should provide helpful and informative responses to the student's questions. Respond in the language of the user.",
      });
      conversation.push({
        role: "assistant",
        content:
          "I am an AI assistant that can help you with your studies. Please ask me any questions you have.",
      });
    }
    // Füge Benutzeranfrage hinzu
    conversation.push({ role: "user", content: transcriptionText });
    // Anfrage an OpenAI
    const aiText = await queryGroq(conversation);
    // Ergebnis anhängen
    conversation.push({ role: "assistant", content: aiText });
    appendToConversationFile(userId, conversation); // Speichert die gesamte Konversation
    // Rückgabe der Ergebnisse
    res.json({ transcription: transcriptionText, aiResponse: aiText });
  } catch (error) {
    console.error(error);
    res.status(500).send(error.message);
  } finally {
    cleanUpFiles([tempInputPath, tempOutputPath]);
  }
});

function appendToConversationFile(userId, conversation) {
  const filename = `./conversations/${userId}.json`;
  // Direktes Schreiben des conversation Arrays als JSON
  fs.writeFile(filename, JSON.stringify(conversation, null, 2), (err) => {
    if (err) {
      console.error("Fehler beim Schreiben in die Datei", err);
    } else {
      console.log("Konversation erfolgreich gespeichert");
    }
  });
}

async function readConversationFile(userId) {
  const filename = `./conversations/${userId}.json`;
  try {
    const data = await readFile(filename, { encoding: "utf8" });
    return JSON.parse(data);
  } catch (err) {
    if (err.code === "ENOENT") {
      // Wenn die Datei nicht existiert, fangen wir frisch an
      return [];
    } else {
      throw err;
    }
  }
}

async function convertAudio(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .output(outputPath)
      .audioCodec("pcm_s16le")
      .audioFrequency(8000)
      .audioChannels(1)
      .on("error", (err) => reject(`Konvertierungsfehler: ${err.message}`))
      .on("end", () => {
        console.log("Konvertierung abgeschlossen.");
        resolve();
      })
      .save(outputPath);
  });
}

async function transcribeAudio(filePath) {
  const audio = await readFile(filePath);
  const audioBase64 = audio.toString("base64");
  const dataUri = `data:audio/wav;base64,${audioBase64}`;

  const output = await replicate.run(
    "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
    { input: { audio: dataUri, batch_size: 64 } }
  );

  console.log(
    `Ergebnis der Spracherkennung: ${JSON.stringify(output, null, 2)}`
  );
  return output.text;
}

async function queryGroq(messages) {
  try {
    const response = await groq.chat.completions.create({
      model: "llama3-8b-8192",
      messages: messages,
    });
    return response.choices[0].message.content;
  } catch (error) {
    console.error("Error in queryGroq:", error);
    throw error; // Weiterleitung des Fehlers zur besseren Fehlerbehandlung
  }
}

async function queryOpenAI(messages) {
  const response = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: messages,
    temperature: 0.8,
    max_tokens: 512,
    top_p: 1,
    frequency_penalty: 0.4,
    presence_penalty: 0.4,
  });

  return response.choices[0].message.content;
}

function cleanUpFiles(paths) {
  paths.forEach((path) => {
    if (fs.existsSync(path)) {
      fs.unlinkSync(path);
    }
  });
}
