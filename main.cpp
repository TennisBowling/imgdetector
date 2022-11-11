#include <iostream>
#include <vector>
#include <string>
#include <cpr/cpr.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <crow.h>
#include <SQLiteCpp/SQLiteCpp.h>

using json = nlohmann::json;

struct Hist
{
    cv::Mat red_hist;
    cv::Mat green_hist;
    cv::Mat blue_hist;

    bool operator==(const Hist &other) const
    {
        return (cv::countNonZero(red_hist != other.red_hist) == 0) && (cv::countNonZero(green_hist != other.green_hist) == 0) && (cv::countNonZero(blue_hist != other.blue_hist) == 0);
    }
};

Hist make_hist(cv::Mat &image)
{
    // make three histograms for red, green and blue
    cv::Mat red_hist, green_hist, blue_hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    int channels[] = {0};
    cv::calcHist(&image, 1, channels, cv::Mat(), red_hist, 1, &histSize, &histRange);
    channels[0] = 1;
    cv::calcHist(&image, 1, channels, cv::Mat(), green_hist, 1, &histSize, &histRange);
    channels[0] = 2;
    cv::calcHist(&image, 1, channels, cv::Mat(), blue_hist, 1, &histSize, &histRange);
    Hist hist;
    hist.red_hist = red_hist;
    hist.green_hist = green_hist;
    hist.blue_hist = blue_hist;
    return hist;
}

int main(int argc, char *argv[])
{

    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] - %v"); // nice style that i like

    crow::SimpleApp app;

    std::vector<Hist> known_images; // the known images are the ones we will check against in the /check endpoint

    SQLite::Database db("./db.sqlite", SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
    db.exec("CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, rawimg TEXT);");

    // get all the images from the database
    SQLite::Statement query(db, "SELECT * FROM images;");

    while (query.executeStep())
    {
        // get the image from the url
        std::string rawimg = query.getColumn("rawimg").getString();
        std::vector<unsigned char> image_data(rawimg.begin(), rawimg.end());
        cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
        cv::Mat hsv;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

        Hist hist = make_hist(hsv);
        known_images.push_back(hist);
    }

    spdlog::info("Loaded {} known images", known_images.size());

    CROW_ROUTE(app, "/set_recognized").methods(crow::HTTPMethod::Post)([&known_images, &db](const crow::request &req)
                                                                       {
        json j = json::parse(req.body);
        cpr::Response r = cpr::Get(cpr::Url{j["url"].get<std::string>()});
        std::vector<unsigned char> image_data(r.text.begin(), r.text.end());

        // load raw image into opencv, convert to hsv, calculate histogram, and store the histogram in the known_images vector
        // do this for red, green, and blue
        cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);

        cv::Mat hsv;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

        Hist hist = make_hist(hsv);

        crow::response res;

        if (std::find(known_images.begin(), known_images.end(), hist) == known_images.end())
        {
            // if the histogram is not in the known_images vector, add it to the database
            SQLite::Statement insert(db, "INSERT INTO images (rawimg) VALUES (?);");
            insert.bind(1, r.text);
            insert.exec();
            known_images.push_back(hist);
            res.code = 200;
            res.body = json({{"success", true}, {"error", "none"}}).dump();
        }
        else
        {
            res.code = 409;
            res.body = json({{"success", false}, {"error", "image_already_recognized"}}).dump();
        }
        return res; });

    CROW_ROUTE(app, "/check").methods(crow::HTTPMethod::Get)([&known_images](const crow::request &req)
                                                             {
        json j = json::parse(req.body);
        cpr::Response r = cpr::Get(cpr::Url{j["url"].get<std::string>()});
        std::vector<unsigned char> image_data(r.text.begin(), r.text.end());
        cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
        cv::Mat hsv;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        Hist h = make_hist(hsv);
        crow::response res;
        // now compare the histogram of the image to the known_images vector

        for (auto &i : known_images)
        {
            double red_dist = cv::compareHist(h.red_hist, i.red_hist, cv::HISTCMP_BHATTACHARYYA);
            double green_dist = cv::compareHist(h.green_hist, i.green_hist, cv::HISTCMP_BHATTACHARYYA);
            double blue_dist = cv::compareHist(h.blue_hist, i.blue_hist, cv::HISTCMP_BHATTACHARYYA);

            if (red_dist < 0.25 && green_dist < 0.25 && blue_dist < 0.25)
            {
                res.code = 200;
                res.body = json({{"result", "match"}, {"distance", red_dist + green_dist + blue_dist}}).dump();
                return res;
            }
        }
        res.code = 200;
        res.body = json({{"result", "no match"}}).dump();
        return res; });

    CROW_ROUTE(app, "/get_recognized").methods(crow::HttpMethod::Get)([&known_images](const crow::request &req)
                                                                     {
        crow::response res;
        res.code = 200;
        // send the known_images vector to the client
        json j;
        for (auto &i : known_images)
        {
            j.push_back(i);
        }
        res.body = j.dump();
        return res
        });

    app.bindaddr("0.0.0.0").port(8000).multithreaded().run();
}