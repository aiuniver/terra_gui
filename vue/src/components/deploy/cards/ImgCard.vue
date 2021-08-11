<template>
  <div class="img-card">
    <img :src="require('@/../public/imgs/img.png')" :alt="ImgAlt">
  </div>
</template>

<script>
export default {
  name: "ImgCard",
  props: {
    ImgUrl: {
      type: String,
      default: '@/../public/imgs/img.png'
    },
    ImgAlt: {
      type: String,
      default: 'image'
    }
  }
}
</script>

<style scoped>
img{
  border: 1px solid #6C7883;
  box-sizing: border-box;
  border-radius: 4px;
  width: 224px;
}
</style>