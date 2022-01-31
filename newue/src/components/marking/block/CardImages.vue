<template>
  <div class="t-card-images">
    <div class="t-card-images__left">
      <div class="t-card-images__row">
        <div class="t-card-images__content">
          <img
            src="https://upload.wikimedia.org/wikipedia/ru/thumb/5/51/%D0%92%D0%B5%D1%81%D1%82%D0%B8_%28%D0%B7%D0%B0%D1%81%D1%82%D0%B0%D0%B2%D0%BA%D0%B0%29.png/640px-%D0%92%D0%B5%D1%81%D1%82%D0%B8_%28%D0%B7%D0%B0%D1%81%D1%82%D0%B0%D0%B2%D0%BA%D0%B0%29.png"
          />
        </div>
      </div>
    </div>
    <div class="t-card-images__center">
      <div class="t-card-images__row">
        <div class="t-card-images__content">
          <img
            src="https://upload.wikimedia.org/wikipedia/ru/thumb/5/51/%D0%92%D0%B5%D1%81%D1%82%D0%B8_%28%D0%B7%D0%B0%D1%81%D1%82%D0%B0%D0%B2%D0%BA%D0%B0%29.png/640px-%D0%92%D0%B5%D1%81%D1%82%D0%B8_%28%D0%B7%D0%B0%D1%81%D1%82%D0%B0%D0%B2%D0%BA%D0%B0%29.png"
          />
        </div>
      </div>
    </div>
    <div class="t-card-images__right">
      <div class="t-card-images__row">
        <div class="t-card-images__content">
          <t-radio @change="change" :lists="lists" />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-card-images',
  components: {
  },
  props: {
    label: String,
    type: String,
  },
  data: () => ({
    lists: [
      {
        label: 'Выходит',
        value: false,
      },
      {
        label: 'Заходит',
        value: true,
      },
    ],
  }),
  computed: {},
  methods: {
    change(e) {
      console.log(e);
    },
  },
};
</script>

<style lang="scss" scoped>
.t-card-images {
  max-width: 490px;
  display: flex;
  border: 1px solid #6c7883;
  box-sizing: border-box;
  border-radius: 4px;
  background: #242f3d;
  height: 190px;
  justify-content: space-between;

  img {
    width: 160px;
    object-fit: cover;
    height: 100%;
    border-radius: 4px;
  }

  &__row {
    padding: 10px;
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  &__right {
    .t-card-images__content {
      display: flex;
      justify-content: center;
      align-items: center;
    }
  }

  &__content {
    flex: 1 0 auto;
  }

  &__footer {
    flex: 0 0 auto;
  }
}
</style>
