<template>
  <div :class="['project', {'project--selected': active}]">
    <div class="project-image">
      <img :src="image" :alt="headline">
      <div class="project-image__edit">
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M0.76274 11.8618C0.587417 11.8615 0.420283 11.7876 0.302115 11.6581C0.181768 11.5296 0.121968 11.3559 0.13774 11.1806L0.290865 9.49685L7.36461 2.4256L9.57524 4.6356L2.50337 11.7062L0.819615 11.8593C0.80024 11.8612 0.780865 11.8618 0.76274 11.8618ZM10.0165 4.19372L7.80649 1.98372L9.13211 0.658097C9.24934 0.540736 9.40842 0.474792 9.5743 0.474792C9.74018 0.474792 9.89926 0.540736 10.0165 0.658097L11.3421 1.98372C11.4595 2.10095 11.5254 2.26003 11.5254 2.42591C11.5254 2.59179 11.4595 2.75087 11.3421 2.8681L10.0171 4.1931L10.0165 4.19372Z" />
          </svg>

          <svg width="7" height="5" viewBox="0 0 7 5" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3.5 4.0625L6.625 0.9375L0.375 0.9375L3.5 4.0625Z"/>
          </svg>
      </div>
    </div>
    <div class="project-detail">
      <div class="project-headline">
        <h3>{{ headline }}</h3>
      </div>
      <div class="project-timespace">
        <div class="project-created">
          <span>{{ created }}</span>
        </div>
        <div class="project-edited">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M2.21 10.2895C2.06974 10.2892 1.93603 10.2301 1.8415 10.1265C1.74522 10.0237 1.69738 9.88471 1.71 9.74445L1.8325 8.39745L7.4915 2.74045L9.26 4.50845L3.6025 10.165L2.2555 10.2875C2.24 10.289 2.2245 10.2895 2.21 10.2895ZM9.613 4.15495L7.845 2.38695L8.9055 1.32645C8.99928 1.23256 9.12654 1.17981 9.25925 1.17981C9.39195 1.17981 9.51921 1.23256 9.613 1.32645L10.6735 2.38695C10.7674 2.48074 10.8201 2.608 10.8201 2.7407C10.8201 2.87341 10.7674 3.00067 10.6735 3.09445L9.6135 4.15445L9.613 4.15495Z" fill="#6C7883"/>
            </svg>

          <span>{{ edited }}</span> 
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CardProject',
  props:['image', 'created', 'edited', 'headline', 'active'],
}
</script>

<style lang="scss" scoped>
.project{
  max-width: 300px;
  border: 1px solid #2B5278;
  border-radius: 4px;
  transition: border .3s ease-in-out;
  cursor: pointer;

  &--selected{
    border: 1px solid #65B9F4;
  }

  &:hover{
    border: 1px solid #65B9F4;
  }

  &-image{
    background: #111111;
    border-radius: 4px;
    height: 170px;
    position: relative;

    &__edit{
      background: #242F3D;
      opacity: 0.5;
      border-radius: 100px;
      width: 45px;
      height: 22px;
      display: flex;
      justify-content: center;
      align-items: center;
      position: absolute;
      top: 10px;
      left: 10px;

      svg{
        fill:#6C7883;
        transition: fill .3s ease-in-out;
        &:first-child{
          margin-right: 7px;
        }
      }
      &:hover{
        svg{
          fill:#F2F5FA;
        }
        opacity: 1;
      }
    }

    img{
      width: 100%;
      object-fit: cover;
      height: 100%;
    }
  }


  &-detail{
    padding: 12px 10px;
  }

  &-headline{
    width: 90%;

    h3{
      font-family: "Open Sans", sans-serif;
      font-weight: normal;
      font-size: 14px;
      white-space: nowrap;
      text-overflow: ellipsis;
      word-wrap: break-word;
    }
  }

  &-timespace{
    margin-top: 7px;
    display: flex;
    justify-content: space-between;
    align-items: center;

    span{
      color: #6C7883;
      font-size: 11px;
    }
  }

  &-edited{
    display: flex;
    align-items: center;

    span{
      margin-left: 5px;
    }
  }
}
</style>