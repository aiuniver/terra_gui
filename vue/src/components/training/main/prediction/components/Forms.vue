<template>
  <div>
    <TableImage v-if="type === 'image'" v-bind="obj" />
    <TableStr v-if="type === 'str' || type === 'number'" v-bind="obj" />
    <TableText v-if="type === 'text'" v-bind="obj" />
    <TableTextSegmented v-if="type === 'segmented_text'" v-bind="obj" />
    <TableVideo v-if="type === 'video'" v-bind="obj" />
    <TableAudio v-if="type === 'audio'" v-bind="obj" />
  </div>
</template>

<script>
export default {
  name: 't-forms-prediction',
  components: {
    TableImage: () => import('./TableImage'),
    TableAudio: () => import('./TableAudio'),
    TableText: () => import('./TableText'),
    TableStr: () => import('./TableStr'),
    TableTextSegmented: () => import('./TableTextSegmented'),
    TableVideo: () => import('./TableVideo.vue'),
  },
  props: {
    type: {
      type: String,
      default: '',
    },
    data: {
      type: Object,
      default: () => {},
    },
    tags_color: {
      type: Object,
      default: () => {},
    },
    layer: {
      type: String,
      default: '',
    },
    update: {
      type: String,
      default: '',
    },
  },
  computed: {
    obj() {
      return { ...this.data, update: this.update, tags_color: this.tags_color, layer: this.layer };
    },
  },
};
</script>

<style lang="scss" scoped>
</style>