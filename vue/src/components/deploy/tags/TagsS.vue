<template>
  <p class="s"><slot>{{ text }}</slot> {{ color }}</p>
</template>

<script>
export default {
  name: ["s1", "s2", "s3", "s4", "s5", "s6"],
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
</style>