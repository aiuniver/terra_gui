<template>
  <div>
    <TableImage v-if="type === 'image'" v-bind="obj" />
    <TableText v-if="type === 'str' || type === 'number'" v-bind="obj" />
    <TableTag v-if="type === 'text'" v-bind="obj" />
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
    TableTag: () => import('./TableTag'),
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
    update: {
      type: String,
      default: '',
    },
  },
  computed: {
    obj() {
      return { ...this.data, update: this.update };
    },
  },
};
</script>

<style lang="scss" scoped>
</style>