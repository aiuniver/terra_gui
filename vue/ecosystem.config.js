module.exports = {
    apps: [
      {
        name: 'test',
        exec_mode: 'cluster',
        instances: 'max',
        script: "./node_modules/@vue/cli-service/bin/vue-cli-service.js",
        args: 'serve'
      }
    ]
  }