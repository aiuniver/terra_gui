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
  border-radius: 4px;
  color: #FFFFFF;
}
p{
  display: inline-block;
}
</style>