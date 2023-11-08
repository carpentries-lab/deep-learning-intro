[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/carpentries-incubator/deep-learning-intro/scaffolds)
[![DOI](https://zenodo.org/badge/163412836.svg)](https://zenodo.org/badge/latestdoi/163412836)


# Introduction to deep learning
This lesson gives an introduction to deep learning.

## Lesson Design
The design of this lesson can be found in the [lesson design](https://carpentries-incubator.github.io/deep-learning-intro/design.html)

## Target Audience
The main audience of this carpentry lesson is PhD students that have little to no experience with
deep learning. In addition, we expect them to know basics of statistics and machine learning.

## Lesson development sprints
We regularly host lesson development sprints, in which we work together at the lesson.
The next one is scheduled for the 30th of November. We kickoff with an online meeting at 9:30 CEST.
If you want to join (you are very welcome to join even if you have never contributed so far) send an email to s.vanderburg@esciencecenter.nl .

## Contributing

We welcome all contributions to improve the lesson! Maintainers will do their best to help you
if you have any questions, concerns, or experience any difficulties along the way.

We'd like to ask you to familiarize yourself with our [Contribution Guide](CONTRIBUTING.md) and
have a look at the [more detailed guidelines][lesson-example] on proper formatting, ways to
render the lesson locally, and even how to write new episodes.

Please see the current list of
[issues](https://github.com/carpentries-incubator/deep-learning_intro/issues)
for ideas for contributing to this repository.

Please also familiarize yourself with the [lesson design](https://carpentries-incubator.github.io/deep-learning-intro/design.html)

For making your contribution, we use the GitHub flow, which is nicely explained in the
chapter [Contributing to a Project](http://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
in Pro Git by Scott Chacon.
Look for the tag ![good_first_issue](https://img.shields.io/badge/-good%20first%20issue-gold.svg).
This indicates that the maintainers will welcome a pull request fixing this issue.

### Teaching this lesson?
We would be very grateful if you can provide us with feedback on this lesson. You can read more about hosting a lesson pilot for an incubator lesson [here](https://docs.carpentries.org/topic_folders/lesson_development/lesson_pilots.html).

You can notify us that you plan to teach this lesson by creating an issue in this repository (and labeling it with beta) or posting a message in the carpentries Slack Machine Learning channel. Please note the questions below to get an indication of the sort of feedback we expect.

After the workshop, lease create an issue (or comment on the issue you created before teaching) with general feedback on teaching the lesson, and label it with beta. As a template, you can use the following questions:
* How much time did you need for the material? (preferably per episode)
* How much time did you need for the exercises?
* Where there any technical issues that arose during setup?
* Where there any bugs or parts of the lesson code that did not work as expected?
* Where there any incorrect or missing exercise solutions?
* Which parts of the lesson were confusing for learners?
* Which questions did learners ask?

In addition, you are very welcome to add issues or pull requests that address more specific feedback.

## Setup the Workshop Website locally

To build this lesson locally, you should follow the [setup instructions for the
workbench](https://carpentries.github.io/sandpaper-docs/#overview). In short,
make sure you have R, Git, and Pandoc installed, open R and use the following
commands to install/update the packages needed for the infrastructure:

```r
# register the repositories for The Carpentries and CRAN
options(repos = c(
  carpentries = "https://carpentries.r-universe.dev/",
  CRAN = "https://cran.rstudio.com/"
))

# Install the template packages to your R library
install.packages(c("sandpaper", "varnish", "pegboard", "tinkr"))
```

## Rendering the website locally
See https://carpentries.github.io/workbench/ for instructions on how to render the website locally.

## Maintainer(s)

Current maintainers of this lesson are
* Peter Steinbach
* Colin Sauze
* Djura Smits
* Sven van der Burg
* Pranav Chandramouli

## Citation and authors

To cite this lesson, please consult with [CITATION.cff](CITATION.cff).
This also holds a list of contributors to the lesson.

[cdh]: https://cdh.carpentries.org
[community-lessons]: https://carpentries.org/community-lessons
[lesson-example]: https://carpentries.github.io/lesson-example
