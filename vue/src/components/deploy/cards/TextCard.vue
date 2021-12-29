<template>
  <div class="text-card">
    <scrollbar :ops="ops">
      <slot></slot>
    </scrollbar>
  </div>
</template>

<script>
export default {
  name: 'TextCard',
  data: () => ({
    ops: {
      scrollPanel: {
        scrollingX: false,
        scrollingY: true,
      },
    },
  }),
};
</script>

<style scoped>
.text-card {
  border: 1px solid #6c7883;
  box-sizing: border-box;
  border-radius: 4px;
  min-height: 74px;
  padding: 6px;
  font-size: 14px;
  line-height: 21px;
  color: #65b9f4;
}
pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: inherit;
}
</style>