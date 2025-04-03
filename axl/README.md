# OHL project week

# How to Fork and Collaborate on This Repository

This repository is designed to help collaborate effectively. Follow these steps to fork the project and contribute:

## Step 1: Fork the Repository
1. Go to the GitHub page of this repository.
2. Click the **Fork** button in the top-right corner of the page.
3. This will create a copy of the repository under your GitHub account.

## Step 2: Clone Your Fork
1. Navigate to your forked repository on GitHub.
2. Click the **Code** button and copy the repository URL.
3. Open a terminal and run the following command to clone the repository to your local machine:
   ```bash
   git clone <your-fork-url>
   ```
4. Replace `<your-fork-url>` with the URL you copied.

## Step 3: Set Up the Upstream Repository
1. Navigate to the cloned repository on your local machine:
   ```bash
   cd <repository-folder>
   ```
2. Add the original repository as the upstream remote:
   ```bash
   git remote add upstream <original-repo-url>
   ```
3. Replace `<original-repo-url>` with the URL of this repository.
4. Verify the remotes:
   ```bash
   git remote -v
   ```

## Step 4: Sync Your Fork with the Upstream Repository
1. Fetch the latest changes from the upstream repository:
   ```bash
   git fetch upstream
   ```
2. Merge the changes into your local main branch:
   ```bash
   git checkout main
   git merge upstream/main
   ```

## Step 5: Make Changes and Commit
1. Create a new branch for your changes:
   ```bash
   git checkout -b <branch-name>
   ```
2. Make your changes and save them.
3. Stage and commit your changes:
   ```bash
   git add .
   git commit -m "<commit-message>"
   ```
4. Replace `<commit-message>` with a meaningful description of your changes.

## Step 6: Push Your Changes
1. Push your branch to your forked repository:
   ```bash
   git push origin <branch-name>
   ```

## Step 7: Create a Pull Request
1. Go to your forked repository on GitHub.
2. Click the **Compare & pull request** button.
3. Add a title and description for your pull request.
4. Click **Create pull request** to submit your changes for review.

## Notes
- Always sync your fork with the upstream repository when you want to see the newest knowlegde.
- You work on your own fork, only if you wrote some code that you want to share with the rest of the groups, you create a pull request to the original repository.
- Make sure to resolve any merge conflicts before creating a pull request.
- Use descriptive commit messages to make it easier for others to understand your changes.
- Be descriptive in your pull request to help reviewers understand your changes.
