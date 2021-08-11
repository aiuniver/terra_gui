<template>
  <div class="text-card">
    {{ text }}
  </div>
</template>

<script>
export default {
  name: "TextCard",
  props: {
    text: {
      type: String,
      default: "text"
    }
  },
}
</script>

<style scoped>
.text-card{
  border: 1px solid #6C7883;
  box-sizing: border-box;
  border-radius: 4px;
  width: 224px;
  min-height: 74px;
  padding: 6px;
  font-size: 14px;
  line-height: 21px;
  color: #65B9F4;
}
</style>