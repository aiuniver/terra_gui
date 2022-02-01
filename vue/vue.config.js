const proxy = {
  bondrogeen: 'http://192.168.1.47:8099/',
  MacBookPro: 'http://localhost:8001/',
  zabastx: 'http://104.248.93.132/',
};
module.exports = {
  // outputDir: "./static/",
  assetsDir: "./static/",
  productionSourceMap: false,
  devServer: {
    proxy: proxy[process.env.USERNAME] || 'http://localhost:8099/',
  },
  configureWebpack: {
    optimization: {
      splitChunks: {
        cacheGroups: {
          default: false,
          // Merge all the CSS into one file
          styles: {
            name: 'styles',
            test: m => m.constructor.name === 'CssModule',
            chunks: 'all',
            minChunks: 1,
            enforce: true
          },
        },
      },
    },
  }
};
