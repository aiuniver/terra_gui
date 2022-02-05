<template>
  <div class="t-colormap">
    <div class="t-colormap__header">{{ graph_name }}</div>
    <div class="t-colormap__body">
      <img width="auto" height="200" :src="src" :alt="'value'" :key="src" />
    </div>
  </div>
</template>

<script>
export default {
  name: 't-colormap',
  props: {
    id: Number,
    task_type: String,
    graph_name: String,
    x_label: String,
    y_label: String,
    plot_data: String,
    data_array: Array,
    labels: Array,
  },
  computed: {
    src() {
      return `/_media/blank/?path=${this.plot_data}&r=${this.id}`;
    },
  },
};
</script>

<style lang="scss" scoped>
.t-colormap {
  position: relative;
  width: 636px;
  height: 352px;
  &__header {
    font-size: 14px;
    line-height: 17px;
    font-weight: 600;
    text-align: center;
    color: #a7bed3;
    margin: 10px auto;
  }
  &__body {
    width: 100%;
    min-height: 300px;
    background-color: #242f3d;
    border: 1px solid #6c7883;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
}
</style>