<template>
  <p class="s"><slot>{{ text }}</slot> {{ color }}</p>
</template>

<script>
export default {
  name: "TagS",
  props: {
    text: {
      type: String
    },
    color: {
      type: String
    }
  }
}
</script>

<style scoped>
.s{
  /*background-color: #0bbc49;*/
  border-radius: 4px;
  color: #FFFFFF;
}
p{
  display: inline-block;
}
</style>